"""Collision-check filter for real-robot joint steps.

Before a policy step is sent to the real UR5e, predict the next joint configuration with
the calibrated kinematic model and check it against the same capsule proxies used in
training (self-collision, table plane, fingertip height). A step that would make things
worse is scaled down (halved repeatedly); if no scaled step is acceptable it is refused.

It is a proxy, not a certified safety function: radii are approximate and the model has
centimetre-level error. It sits in front of, and does not replace, the controller's own
limits, the safety_mode watchdog and the e-stop. Off by default in the deployment script
(--safety-filter). The table height in the robot base frame must be measured: it is 0.0
when the arm is mounted on the table top, as in simulation.
"""
import numpy as np

from ur5e_grasp import kinematics as K


class SafetyFilter:
    def __init__(self, table_z_base=0.0, table_margin_m=0.02, self_margin_m=0.01, tip_margin_m=0.0, min_scale=0.125):
        self.table_z_base = table_z_base
        self.table_margin_m = table_margin_m
        self.self_margin_m = self_margin_m
        self.tip_margin_m = tip_margin_m
        self.min_scale = min_scale

    def measure(self, q, grip=0.0):
        s, _ = K.self_clearance(q, grip)
        t = K.table_clearance(q, grip, table_z_base=self.table_z_base, include_tip=False)
        tip = float(K.tool_point_base(q, grip)[2]) - self.table_z_base
        return s, t, tip

    def filter(self, q_now, delta, grip=0.0):
        """Return (delta_out, note). delta_out is None when the step must not be executed."""
        q_now = np.asarray(q_now, dtype=float)
        delta = np.asarray(delta, dtype=float)
        s0, t0, tip0 = self.measure(q_now, grip)

        def acceptable(q):
            s, t, tip = self.measure(q, grip)
            return (
                (s >= self.self_margin_m or s >= s0)
                and (t >= self.table_margin_m or t >= t0)
                and (tip >= self.tip_margin_m or tip >= tip0)
            )

        scale = 1.0
        while scale >= self.min_scale:
            if acceptable(q_now + delta * scale):
                return delta * scale, ("" if scale == 1.0 else f"step scaled to {scale:g}")
            scale *= 0.5
        return None, "predicted self-collision or table contact"
