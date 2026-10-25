"""Collision queries for planning, evaluated in a separate copy of the scene."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contacts import CollisionMode, GeomGroups, allowed_in, contact_labels
from .scene import clone_query_data

StateWriter = Callable[[object, object, NDArray[np.float64]], None]


def planning_model(model, clearance_m: float):
    """Copy the model with every collision margin set to the planning clearance.

    MuJoCo then reports geom pairs closer than clearance_m as contacts, so one
    contact pass enforces the clearance.
    """
    planning = copy.deepcopy(model)
    active = (np.asarray(planning.geom_contype) != 0) | (
        np.asarray(planning.geom_conaffinity) != 0
    )
    planning.geom_margin[active] = float(clearance_m)
    return planning


class CollisionQuery:
    """Check arm configurations for one planning state in a private MjData."""

    def __init__(
        self,
        model,
        groups: GeomGroups,
        write_state: StateWriter,
        *,
        source_data,
        clearance_m: float,
        configuration_is_valid: Callable[[NDArray[np.float64]], bool] | None = None,
    ):
        import mujoco

        self.model = model
        self.groups = groups
        self.write_state = write_state
        self.data = clone_query_data(model, source_data)
        self.clearance_m = float(clearance_m)
        self.configuration_is_valid = configuration_is_valid
        self._forward = mujoco.mj_forward

    def labels(self, q_arm: ArrayLike) -> set[str]:
        q = np.asarray(q_arm, dtype=np.float64)
        self.write_state(self.model, self.data, q.copy())
        self._forward(self.model, self.data)
        return contact_labels(
            self.model, self.data, self.groups, clearance_m=self.clearance_m
        )

    def is_valid(self, q_arm: ArrayLike, mode: CollisionMode) -> bool:
        q = np.asarray(q_arm, dtype=np.float64)
        if self.configuration_is_valid is not None and not self.configuration_is_valid(q):
            return False
        return allowed_in(self.labels(q), mode)

    def predicate(self, mode: CollisionMode) -> Callable[[NDArray[np.float64]], bool]:
        def state_is_valid(q: NDArray[np.float64]) -> bool:
            return self.is_valid(q, mode)

        return state_is_valid
