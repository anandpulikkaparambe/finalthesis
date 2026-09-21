"""UR5e forward kinematics, a capsule collision proxy, and a damped-least-squares IK step.

Pure numpy, no Isaac Sim dependency, so it is unit-testable anywhere and is shared by
the training environment (collision proxies), the demonstration controller, and the
real-robot safety filter.

DH parameters are the Universal Robots published UR5e values. The base pose and the
tool offset were fitted to the headline run's telemetry (joint angles vs. the logged
world-frame finger-pad midpoint): median position error 3.7 mm, 95th percentile 2 cm.
Treat the collision proxy as approximate: radii are hand-set capsules, not the meshes.
"""
import numpy as np

# Standard DH: A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# Fitted to telemetry (see analysis/fit_kinematics.py). World frame of the Isaac scene.
BASE_POS_WORLD = np.array([0.0015, 0.5517, 0.7797])
BASE_YAW = np.pi  # UR base_link is rotated 180 deg about z relative to the world frame
TOOL_OFFSET = np.array([-0.0008, -0.0131, 0.1787])          # flange frame -> finger-pad midpoint, gripper open
TOOL_OFFSET_PER_GRIP = np.array([0.0023, 0.0982, -0.0152])  # change per unit of finger_joint position

# Capsule radii (m) for the collision proxy. Approximate on purpose.
RADIUS_SHOULDER = 0.075
RADIUS_UPPER = 0.060
RADIUS_FOREARM = 0.055
RADIUS_WRIST = 0.04
RADIUS_TOOL = 0.065   # Robotiq 2F-140 body
RADIUS_TOOL_TABLE = 0.03  # gripper half-thickness used against the table plane (the body is wider than it is thick)


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward_chain(q):
    """Return the 7 transforms T_0_i (i = 0..6) in the robot base frame."""
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    chain = [T.copy()]
    for i in range(6):
        T = T @ _dh(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
        chain.append(T.copy())
    return chain


def tool_point_base(q, grip=0.0, chain=None):
    """Finger-pad midpoint in the base frame."""
    chain = chain if chain is not None else forward_chain(q)
    T = chain[6]
    offset = TOOL_OFFSET + float(grip) * TOOL_OFFSET_PER_GRIP
    return T[:3, 3] + T[:3, :3] @ offset


def tool_axis_base(q, chain=None):
    """Unit approach axis (flange z) in the base frame."""
    chain = chain if chain is not None else forward_chain(q)
    return chain[6][:3, 2].copy()


def base_to_world(p_base, base_pos=BASE_POS_WORLD, yaw=BASE_YAW):
    return _rot_z(yaw) @ np.asarray(p_base, dtype=float) + base_pos


def world_to_base(p_world, base_pos=BASE_POS_WORLD, yaw=BASE_YAW):
    return _rot_z(yaw).T @ (np.asarray(p_world, dtype=float) - base_pos)


def tool_point_world(q, grip=0.0):
    return base_to_world(tool_point_base(q, grip))


def position_jacobian(q, grip=0.0, eps=1e-5):
    """3x6 numerical Jacobian of the tool point (base frame)."""
    q = np.asarray(q, dtype=float)
    J = np.zeros((3, 6))
    p0 = tool_point_base(q, grip)
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (tool_point_base(dq, grip) - p0) / eps
    return J


def angular_jacobian(q):
    """3x6 angular-velocity Jacobian of the tool frame (base frame): joint axes z_{i-1}."""
    chain = forward_chain(q)
    return np.column_stack([chain[i][:3, 2] for i in range(6)])


def dls_step(q, target_base, grip=0.0, damping=0.05, max_step=0.08, q_ref=None, nullspace_gain=0.2,
             axis_target=None, axis_gain=0.5):
    """One damped-least-squares step towards `target_base` (tool point, base frame).

    If `axis_target` (unit vector, base frame) is given, the approach axis is part of the
    main task: the stacked position + axis-direction error is solved together, weighted by
    `axis_gain`. A posture pull towards `q_ref` acts in the remaining null space. Returns a
    joint delta clipped element-wise to +-max_step.
    """
    q = np.asarray(q, dtype=float)
    pos_err = np.asarray(target_base, dtype=float) - tool_point_base(q, grip)
    Jp = position_jacobian(q, grip)
    if axis_target is None:
        J, err = Jp, pos_err
    else:
        axis_now = tool_axis_base(q)
        rot_err = np.cross(axis_now, np.asarray(axis_target, dtype=float))
        J = np.vstack([Jp, axis_gain * angular_jacobian(q)])
        err = np.concatenate([pos_err, axis_gain * rot_err])
    JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
    J_pinv = J.T @ np.linalg.inv(JJt)
    dq = J_pinv @ err
    if q_ref is not None:
        N = np.eye(6) - J_pinv @ J
        dq = dq + N @ (nullspace_gain * (np.asarray(q_ref, dtype=float) - q))
    return np.clip(dq, -max_step, max_step)

# ----------------------------------------------------------------------------- collision proxy

def _segment_distance(p1, q1, p2, q2):
    """Minimum distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def capsules(q, grip=0.0):
    """Arm capsules in the base frame: list of (name, p_start, p_end, radius)."""
    chain = forward_chain(q)
    o = [T[:3, 3] for T in chain]
    tip = tool_point_base(q, grip, chain)
    return [
        ("shoulder", o[0], o[1], RADIUS_SHOULDER),
        ("upper_arm", o[2], o[3], RADIUS_UPPER),
        ("forearm", o[3], o[4], RADIUS_FOREARM),
        ("wrist1", o[4], o[5], RADIUS_WRIST),
        ("wrist2", o[5], o[6], RADIUS_WRIST),
        ("tool", o[6], tip, RADIUS_TOOL),
    ]


# Capsule pairs that can plausibly collide. Adjacent links and the compact wrist pairs
# (forearm-wrist2, wrist1-tool) touch or overlap by design and are excluded.
_SELF_PAIRS = [
    ("shoulder", "forearm"), ("shoulder", "wrist1"), ("shoulder", "wrist2"), ("shoulder", "tool"),
    ("upper_arm", "wrist1"), ("upper_arm", "wrist2"), ("upper_arm", "tool"),
    ("forearm", "tool"),
]


def self_clearance(q, grip=0.0):
    """Smallest signed clearance (m) over non-adjacent capsule pairs; negative = overlap."""
    caps = {name: (p, qq, r) for name, p, qq, r in capsules(q, grip)}
    best, best_pair = np.inf, None
    for a, b in _SELF_PAIRS:
        pa, qa, ra = caps[a]
        pb, qb, rb = caps[b]
        c = _segment_distance(pa, qa, pb, qb) - ra - rb
        if c < best:
            best, best_pair = c, (a, b)
    return float(best), best_pair


def table_clearance(q, grip=0.0, table_z_base=0.0, finger_radius=0.01, include_tip=True):
    """Smallest clearance (m) between the table plane and the arm/tool; negative = below it.

    The plane is at `table_z_base` in the base frame (the robot is mounted on the table
    top, so 0.0). The shoulder capsule is excluded because it legitimately sits on the
    table. The tool tip uses a small finger radius so that touching the target on the
    table is not counted as a violation; pass include_tip=False to ignore the tip entirely.
    """
    best = np.inf
    for name, p, qq, r in capsules(q, grip):
        if name == "shoulder":
            continue
        if name == "tool":
            best = min(best, p[2] - RADIUS_TOOL_TABLE - table_z_base)
            if include_tip:
                best = min(best, qq[2] - finger_radius - table_z_base)
        else:
            best = min(best, min(p[2], qq[2]) - r - table_z_base)
    return float(best)
