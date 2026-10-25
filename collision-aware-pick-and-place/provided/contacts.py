"""Classify MuJoCo contacts and apply the pick-and-place contact rules.

Every contact is labeled by the two geom groups it joins. Each phase allows a
fixed set of labels; any other contact is a collision.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


COLLISION_PENETRATION_TOLERANCE_M = 5e-5


class CollisionMode(str, Enum):
    """Manipulation phases with different contact rules."""

    FREE = "free"  # transit, approach, retreat
    CLOSURE = "closure"  # fingers close on the resting target
    LIFT = "lift"  # held target may still touch the support
    CARRY = "carry"  # held target must clear support and obstacles
    PLACE = "place"  # held target may touch the support


# Contact labels permitted in each phase. The resting target always touches the
# support except while it is carried clear of it.
ALLOWED_CONTACTS: dict[CollisionMode, frozenset[str]] = {
    CollisionMode.FREE: frozenset({"target-support"}),
    CollisionMode.CLOSURE: frozenset({"fingers-target", "target-support"}),
    CollisionMode.LIFT: frozenset({"fingers-target", "target-support"}),
    CollisionMode.CARRY: frozenset({"fingers-target"}),
    CollisionMode.PLACE: frozenset({"fingers-target", "target-support"}),
}


@dataclass(frozen=True)
class ContactRecord:
    geom1: int
    geom2: int
    distance_m: float
    position_world: NDArray[np.float64]


@dataclass(frozen=True)
class GeomGroups:
    """Collision geom ids for each part of the scene."""

    base: frozenset[int]
    arm: frozenset[int]
    fingers: frozenset[int]
    target: frozenset[int]
    support: frozenset[int]
    obstacles: frozenset[int]

    @property
    def robot(self) -> frozenset[int]:
        return self.base | self.arm | self.fingers

    @property
    def environment(self) -> frozenset[int]:
        return self.support | self.obstacles


def _collision_geoms(model) -> NDArray[np.int64]:
    active = (np.asarray(model.geom_contype) != 0) | (
        np.asarray(model.geom_conaffinity) != 0
    )
    return np.flatnonzero(active)


def geom_ids_for_bodies(model, body_names: Iterable[str]) -> frozenset[int]:
    """Collision geoms attached to the named bodies."""
    body_ids = set()
    for name in body_names:
        body_id = int(model.body(str(name)).id)
        if body_id < 0:
            raise ValueError(f"unknown MuJoCo body {name!r}")
        body_ids.add(body_id)
    bodies = np.asarray(model.geom_bodyid)
    return frozenset(int(g) for g in _collision_geoms(model) if bodies[g] in body_ids)


def support_geom_ids(model, support_names: Iterable[str]) -> frozenset[int]:
    """Collision geoms whose own or body name exactly matches a support name."""
    import mujoco

    names = {str(name).lower() for name in support_names}
    support = set()
    for g in _collision_geoms(model):
        geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        body = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])
            )
            or ""
        )
        if geom.lower() in names or body.lower() in names:
            support.add(int(g))
    return frozenset(support)


def scene_geom_groups(
    model,
    *,
    target_body_name: str,
    base_body_names: Iterable[str],
    arm_body_names: Iterable[str],
    finger_body_names: Iterable[str],
    support_names: Iterable[str],
) -> GeomGroups:
    """Robot, target, and support by name; every other collision geom is an obstacle.

    Static geometry that is not named as support (a fixed wall, say) is an
    obstacle, so the target may not touch it.
    """
    base = geom_ids_for_bodies(model, base_body_names)
    arm = geom_ids_for_bodies(model, arm_body_names)
    fingers = geom_ids_for_bodies(model, finger_body_names)
    target = geom_ids_for_bodies(model, (target_body_name,))
    if not target:
        raise RuntimeError("target collision geometry could not be identified")
    collision = frozenset(int(g) for g in _collision_geoms(model))
    support = support_geom_ids(model, support_names) - base - arm - fingers - target
    if not support:
        raise RuntimeError(f"no support geometry matches {tuple(support_names)!r}")
    obstacles = collision - base - arm - fingers - target - support
    return GeomGroups(base, arm, fingers, target, support, obstacles)


def read_contacts(
    model, data, *, maximum_distance_m: float = 0.0
) -> list[ContactRecord]:
    """Contacts whose signed distance is at most maximum_distance_m."""
    records = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) > maximum_distance_m:
            continue
        records.append(
            ContactRecord(
                int(contact.geom1),
                int(contact.geom2),
                float(contact.dist),
                np.array(contact.pos, dtype=np.float64),
            )
        )
    return records


def contact_label(a: int, b: int, groups: GeomGroups) -> str | None:
    """Name the kind of contact between geoms a and b; None if it never matters."""

    def pair(left, right):
        return (a in left and b in right) or (b in left and a in right)

    if a in groups.robot and b in groups.robot:
        return "self"
    if pair(groups.base, groups.environment):
        return None  # the base is mounted on the support
    if pair(groups.arm | groups.fingers, groups.environment):
        return "robot-environment"
    if pair(groups.fingers, groups.target):
        return "fingers-target"
    if pair(groups.arm | groups.base, groups.target):
        return "robot-target"
    if pair(groups.target, groups.support):
        return "target-support"
    if pair(groups.target, groups.obstacles):
        return "target-obstacle"
    if a in groups.environment and b in groups.environment:
        return None
    return "unknown"


def contact_labels(model, data, groups: GeomGroups, *, clearance_m: float) -> set[str]:
    """Labels of all geom pairs closer than clearance_m.

    A negative clearance tolerates that much penetration. Robot self-contact is
    counted only on penetration, since neighboring links are always close.
    """
    labels = set()
    for contact in read_contacts(model, data, maximum_distance_m=clearance_m):
        label = contact_label(contact.geom1, contact.geom2, groups)
        if label is None:
            continue
        if label == "self" and contact.distance_m > min(clearance_m, 0.0):
            continue
        labels.add(label)
    return labels


def allowed_in(labels: set[str], mode: CollisionMode | str) -> bool:
    return labels <= ALLOWED_CONTACTS[CollisionMode(mode)]


def _sensed_contact_pairs(data):
    """Geom pairs touching geometrically or included in MuJoCo's contact solver."""
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        if float(contact.dist) <= 0.0 or int(contact.efc_address) >= 0:
            yield int(contact.geom1), int(contact.geom2)


def target_finger_contacts(model, data, groups, finger_groups) -> tuple[bool, bool]:
    """Whether each finger touches the target."""
    touched = [False, False]
    for a, b in _sensed_contact_pairs(data):
        for index, finger in enumerate(finger_groups):
            if (a in finger and b in groups.target) or (
                b in finger and a in groups.target
            ):
                touched[index] = True
    return touched[0], touched[1]


def target_environment_contacts(model, data, groups) -> tuple[bool, bool]:
    """Whether the target touches the support and whether it touches an obstacle."""
    labels = {
        contact_label(a, b, groups) for a, b in _sensed_contact_pairs(data)
    }
    return "target-support" in labels, "target-obstacle" in labels
