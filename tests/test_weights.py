"""Unit tests for relevance resolution and the penalty (no simulator needed).

    python -m pytest relevance/tests -q
"""

import glob
import os

import pytest

from relevance import weights
from relevance.env import episode_metrics
from relevance.weights import CONFIG_DIR, RelevanceConfig, damage_penalty, load_relevance_config


class DamageableMJCFObject:
    def __init__(self, name):
        self.name = name


class DamageableCounter(DamageableMJCFObject):
    pass


class DamageableSink(DamageableMJCFObject):
    pass


class DamageableWall(DamageableMJCFObject):
    pass


class DamageableWheeledRobot(DamageableMJCFObject):
    pass


class FakeEnv:
    def __init__(self, objects, sink):
        self.objects = objects
        self.sink = sink

    def _is_robot(self, obj):
        return isinstance(obj, DamageableWheeledRobot)


# Distances from the anchors used by the fake geometry below.
DIST = {"island_counter": 0.0, "dishwasher": 0.2, "wall_left": 2.5, "wall_right": 2.0,
        "fridge": 1.8, "PandaOmron": 0.1}


@pytest.fixture
def scene(monkeypatch):
    robot = DamageableWheeledRobot("PandaOmron")
    bowl, cup = DamageableMJCFObject("bowl"), DamageableMJCFObject("cup")
    sink = DamageableSink("sink_island_group")
    counter = DamageableCounter("island_counter")
    dishwasher = DamageableMJCFObject("dishwasher")
    walls = [DamageableWall("wall_left"), DamageableWall("wall_right")]
    fridge = DamageableMJCFObject("fridge")
    objects = [robot, bowl, cup, sink, counter, dishwasher, fridge] + walls

    def fake_distance(env, a, b):
        if b.name in DIST:
            return DIST[b.name]
        return DIST.get(a.name, 0.0)

    monkeypatch.setattr(weights, "object_distance", fake_distance)
    return FakeEnv(objects, sink), objects


def cfg(**raw):
    return RelevanceConfig.from_dict(raw, name="test")


def test_name_ref_robot_type_and_default(scene):
    env, objects = scene
    rho = cfg(default_relevance=0.1, relevance={
        "robot": 1.0, "bowl": 0.9, "ref:sink": 0.75, "type:wall": 0.25, "contains:wash": 0.5,
    }).resolve(env, objects)
    assert rho["PandaOmron"] == 1.0
    assert rho["bowl"] == 0.9
    assert rho["sink_island_group"] == 0.75
    assert rho["wall_left"] == rho["wall_right"] == 0.25
    assert rho["dishwasher"] == 0.5
    assert rho["cup"] == rho["fridge"] == 0.1


def test_more_specific_key_wins(scene):
    env, objects = scene
    rho = cfg(relevance={"type:wall": 0.25, "wall_left": 1.0, "contains:wall": 0.5}).resolve(env, objects)
    assert rho["wall_left"] == 1.0      # name beats type and contains
    assert rho["wall_right"] == 0.25    # type beats contains


def test_unmatched_key_warns(scene, capsys):
    env, objects = scene
    cfg(relevance={"bowll": 1.0}).resolve(env, objects)
    assert "bowll" in capsys.readouterr().out


def test_nearby_rule(scene):
    env, objects = scene
    rho = cfg(relevance={"robot": 1.0, "bowl": 1.0, "ref:sink": 1.0},
              nearby=[{"of": ["bowl", "ref:sink"], "radius": 0.3, "relevance": 0.5}]).resolve(env, objects)
    assert rho["island_counter"] == rho["dishwasher"] == 0.5
    assert rho["PandaOmron"] == 1.0     # explicit key beats nearby
    assert rho["wall_left"] == rho["fridge"] == 0.0


def test_far_size_matched_has_same_size_as_near(scene):
    env, objects = scene
    base = {"robot": 1.0, "bowl": 1.0, "cup": 1.0, "ref:sink": 1.0}
    near = cfg(relevance=base, nearby=[{"of": ["bowl", "ref:sink"], "radius": 0.3}]).resolve(env, objects)
    far = cfg(relevance=base, nearby=[{"of": ["bowl", "ref:sink"], "radius": 0.3,
                                       "select": "far_size_matched", "min_distance": 1.5}]).resolve(env, objects)
    explicit = {"PandaOmron", "bowl", "cup", "sink_island_group"}
    added_near = {n for n, r in near.items() if r > 0} - explicit
    added_far = {n for n, r in far.items() if r > 0} - explicit
    assert len(added_near) == len(added_far) == 2
    assert added_far <= {"wall_left", "wall_right", "fridge"}


def test_relevance_out_of_range():
    with pytest.raises(ValueError):
        cfg(relevance={"bowl": 1.5})


def test_penalty_math():
    prev = {"a": 100.0, "b": 100.0, "c": 50.0}
    now = {"a": 90.0, "b": 100.0, "c": 40.0}
    penalty, damage = damage_penalty(prev, now, {"a": 1.0, "c": 0.5}, weight=0.1)
    assert damage == {"a": 10.0, "b": 0.0, "c": 10.0}
    assert penalty == pytest.approx(0.1 * (10.0 + 0.5 * 10.0))
    # All-zero relevance means no penalty, whatever the damage.
    assert damage_penalty(prev, now, {}, weight=0.1)[0] == 0.0


def test_episode_metrics_safe_success_uses_every_object():
    totals = {"task_return": 1.0, "damage_penalty": 0.0, "in_scope_damage": 0.0, "out_of_scope_damage": 6.0}
    m = episode_metrics({"cup": 100.0, "laptop": 94.0}, {"cup": 1.0}, True, False, 10, totals)
    assert m["success"] and not m["safe_success"]   # out-of-scope laptop still counts for safety
    assert m["weighted_health"] == 100.0
    assert m["num_damaged_objects"] == 1


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(CONFIG_DIR, "*", "*.yaml"))))
def test_shipped_configs_load(path):
    task = os.path.basename(os.path.dirname(path))
    load_relevance_config(path, task=task)
