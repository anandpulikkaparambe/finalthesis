import json

import numpy as np

from ur5e_grasp import spec
from ur5e_grasp.config import ContactConfig, CurriculumConfig, EnvConfig, config_from_dict, load_config, save_config
from ur5e_grasp.contact import HoldTracker, gap_proxy_contact, pads_in_contact
from ur5e_grasp.curriculum import CurriculumManager
from dataclasses import asdict


def _feed(cm, rate, n, steps):
    for i in range(n):
        cm.record_episode(i < int(rate * n))
    return cm.update(steps)


def test_curriculum_advances_on_success_and_regresses_on_failure():
    cm = CurriculumManager(CurriculumConfig(warmup_steps=0))
    lvl = _feed(cm, 1.0, 20, 100000)
    assert np.isclose(lvl, 0.05)
    lvl = _feed(cm, 0.0, 20, 100000)
    assert np.isclose(lvl, 0.0)


def test_curriculum_warmup_cap_and_floor():
    cm = CurriculumManager(CurriculumConfig(warmup_steps=50000, warmup_cap=0.1))
    for _ in range(10):
        _feed(cm, 1.0, 20, 1000)
    assert cm.level <= 0.1 + 1e-9
    cm2 = CurriculumManager(CurriculumConfig(initial_level=0.3, warmup_steps=0))
    _feed(cm2, 0.0, 20, 100000)
    assert cm2.level >= 0.3 - 1e-9


def test_hold_tracker_needs_consecutive_steps():
    ht = HoldTracker(ContactConfig(hold_steps=3))
    ht.reset(0.8)
    assert not ht.update(True, True, True)
    assert not ht.update(True, True, True)
    assert ht.update(True, True, True)
    ht.update(False, True, True)          # contact lost
    assert ht.count == 0


def test_hold_tracker_lift_requirement():
    ht = HoldTracker(ContactConfig(hold_steps=1, lift_required_m=0.05))
    ht.reset(0.8)
    assert not ht.update(True, True, True, target_height=0.81)
    assert ht.update(True, True, True, target_height=0.86)


def test_pad_contact_needs_both_pads():
    cfg = ContactConfig(min_pad_force_n=0.5)
    assert pads_in_contact(1.0, 1.0, cfg)
    assert not pads_in_contact(1.0, 0.1, cfg)


def test_gap_proxy_matches_original_definition():
    cfg = ContactConfig()
    assert gap_proxy_contact(0.8, 0.5, 0.9, cfg)
    assert not gap_proxy_contact(0.8, 0.79, 0.9, cfg)
    assert not gap_proxy_contact(0.8, 0.5, 0.2, cfg)


def test_config_roundtrip(tmp_path):
    cfg = EnvConfig()
    cfg.reward.velocity_penalty_coef = 0.02
    cfg.randomization.kp_scale_range = (0.5, 1.5)
    p = tmp_path / "cfg.json"
    save_config(cfg, str(p))
    back = load_config(str(p))
    assert back.reward.velocity_penalty_coef == 0.02
    assert back.randomization.kp_scale_range == (0.5, 1.5)
    assert asdict(back) == asdict(cfg)


def test_partial_config_uses_defaults():
    cfg = config_from_dict({"reward": {"velocity_penalty_coef": 0.01}, "self_collision": False})
    assert cfg.reward.velocity_penalty_coef == 0.01 and cfg.self_collision is False
    assert cfg.contact.hold_steps == ContactConfig().hold_steps


def test_spec_gripper_mapping():
    assert spec.gripper_closed_amount(-np.pi) == 0.0 and spec.gripper_closed_amount(np.pi) == 1.0
    assert np.isclose(spec.gripper_cmd(np.pi), 0.8)
    assert spec.neutral_action().shape == (spec.ACTION_DIM,)