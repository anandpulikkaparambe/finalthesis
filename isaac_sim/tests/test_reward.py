import numpy as np

from ur5e_grasp.config import RewardConfig
from ur5e_grasp.reward import StepQuantities, TerminalInputs, apply_floor, classify_terminal, dense_reward

LO = np.full(6, -6.0)
HI = np.full(6, 6.0)


def _q(**kw):
    base = dict(prev_dist=1.0, dist=0.9, joint_vel=np.zeros(6), joint_pos=np.zeros(6),
                joint_pos_min=LO, joint_pos_max=HI)
    base.update(kw)
    return StepQuantities(**base)


def test_progress_is_rewarded_and_regress_is_punished():
    cfg = RewardConfig()
    assert dense_reward(_q(), cfg)[0] > 0
    assert dense_reward(_q(prev_dist=0.9, dist=1.0), cfg)[0] < 0


def test_velocity_penalty_uses_configured_coefficient():
    vel = np.full(6, 2.0)
    r_small, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, joint_vel=vel), RewardConfig(velocity_penalty_coef=0.001, energy_penalty_coef=0.0))
    r_large, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, joint_vel=vel), RewardConfig(velocity_penalty_coef=0.05, energy_penalty_coef=0.0))
    assert np.isclose(r_small, -0.001 * 12.0) and np.isclose(r_large, -0.05 * 12.0)


def test_energy_penalty_prefers_effort_signal():
    cfg = RewardConfig(velocity_penalty_coef=0.0, energy_penalty_coef=0.1)
    with_effort, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, joint_vel=np.ones(6), efforts=np.full(6, 10.0)), cfg)
    assert np.isclose(with_effort, -0.1 * 60.0 * 0.1)


def test_collision_clearances_penalise_only_inside_margin():
    cfg = RewardConfig()
    far, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, self_clearance=0.5, table_clearance=0.5), cfg)
    near, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, self_clearance=0.0, table_clearance=0.0), cfg)
    assert far == 0.0 and near < 0.0


def test_joint_limit_penalty():
    cfg = RewardConfig()
    mid, _ = dense_reward(_q(prev_dist=1.0, dist=1.0), cfg)
    edge, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, joint_pos=np.array([5.95, 0, 0, 0, 0, 0])), cfg)
    assert edge < mid


def test_contact_reward_uses_weaker_pad_not_sum():
    # min(), not sum/either -- a single pad touching just slides a light target along that one
    # pad without trapping it (confirmed live, demo_controller.py's own act() comment), so only
    # BOTH pads engaged should earn anything.
    cfg = RewardConfig(contact_coef=0.1, contact_confirm_force_n=0.5)
    one_pad, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, contact_force_l=5.0, contact_force_r=0.0), cfg)
    both_pads, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, contact_force_l=0.5, contact_force_r=0.5), cfg)
    assert np.isclose(one_pad, 0.0)
    assert np.isclose(both_pads, 0.1)


def test_contact_reward_clips_past_confirm_threshold():
    # no incentive to squeeze harder once both pads are past the confirm threshold -- confirmed
    # live that extra squeeze on this light, low-friction target increases displacement, not grip.
    cfg = RewardConfig(contact_coef=0.1, contact_confirm_force_n=0.5)
    at_threshold, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, contact_force_l=0.5, contact_force_r=0.5), cfg)
    way_past, _ = dense_reward(_q(prev_dist=1.0, dist=1.0, contact_force_l=50.0, contact_force_r=50.0), cfg)
    assert np.isclose(at_threshold, way_past) and np.isclose(at_threshold, 0.1)


def test_floor_clamps_episode_sum():
    r, s = apply_floor(-3.0, -4.0, -5.0)
    assert s == -5.0 and np.isclose(r, -1.0)
    r, s = apply_floor(-0.5, -1.0, -5.0)
    assert r == -0.5 and s == -1.5


def _t(**kw):
    base = dict(grasp_confirmed=False, false_grasp=False, joint_vel=np.zeros(6),
                table_collision=False, self_collision=False, target_lost=False, dist_to_target=1.0)
    base.update(kw)
    return TerminalInputs(**base)


def test_success_requires_confirmed_grasp():
    cfg = RewardConfig()
    done, delta, reason = classify_terminal(_t(grasp_confirmed=True), cfg)
    assert done and delta == 15.0 and reason == "Full Task Success"
    done, delta, reason = classify_terminal(_t(false_grasp=True), cfg)
    assert done and delta == -5.0 and "False Grasp" in reason


def test_no_termination_without_events():
    assert classify_terminal(_t(), RewardConfig())[0] is False


def test_self_collision_terminates_and_overrides_success():
    done, delta, reason = classify_terminal(_t(grasp_confirmed=True, self_collision=True), RewardConfig())
    assert done and reason == "Self Collision" and delta == -5.0


def test_self_collision_can_be_made_non_terminal():
    cfg = RewardConfig(self_collision_terminates=False)
    assert classify_terminal(_t(self_collision=True), cfg)[0] is False


def test_table_collision_penalty_scales_with_distance():
    cfg = RewardConfig()
    _, near, _ = classify_terminal(_t(table_collision=True, dist_to_target=0.0), cfg)
    _, far, _ = classify_terminal(_t(table_collision=True, dist_to_target=0.3), cfg)
    assert np.isclose(near, -5.0) and np.isclose(far, -10.0)


def test_kill_switch():
    done, _, reason = classify_terminal(_t(joint_vel=np.array([11.0, 0, 0, 0, 0, 0])), RewardConfig())
    assert done and "Kill-Switch" in reason