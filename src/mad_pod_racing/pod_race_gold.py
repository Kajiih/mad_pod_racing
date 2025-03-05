"""
Mad Pod Race Gold.

Todo:
Runner:
- Find the next target point depending on:
    - position of the current checkpoint
    - position of the next checkpoint
        - with apexing (consider the radius of the cp)
    - position of the pod
    - velocity of the pod
    - Improvement:
        - Position of the opponent hunter pod (avoid)
    - Trajectory interpolation:
        - Model Predictive Control (MPC)
        - Clothoids/ G2 Hermite Interpolation: https://thegraphicsblog.com/2018/04/30/clothoids-the-perfect-curve/
        - Catmull-Rom Splines: https://qroph.github.io/2018/07/30/smooth-paths-using-catmull-rom-splines.html, arc length: https://gamedev.stackexchange.com/questions/14985/determine-arc-length-of-a-catmull-rom-spline-to-move-at-a-constant-speed
        - OGH curves: https://math.stackexchange.com/questions/1021381/how-can-i-limit-the-amount-of-curvature-of-a-bezier-curve
        - Cubic Splines/Cubic Hermite spline
- Chose when to boost
    - Distance to the current to large enough (pod doesn't arrive to fast/has time to slow down before the cp if necessary) and angle close to target point
    - Aligned with last cp
- Chose when to use shield
    - When a collision will happen at the next turn and the bounce will push the pod far from it's trajectory

Implement opti algo:
- https://www.codingame.com/blog/genetic-algorithms-coders-strike-back-game/
- https://www.codingame.com/blog/evolutionary-trajectory-optimization/
Check: https://files.magusgeek.com/csb/csb_en.html // https://files.magusgeek.com/csb/csb.html
"""

from __future__ import annotations

import math
import sys
from abc import ABC
from enum import StrEnum
from functools import cached_property
from math import exp
from typing import Any, Literal

import numpy as np
from numpy import array, asarray, bool_, dtype, float64, floating, int_, ndarray
from numpy.polynomial import Polynomial
from numpy.typing import NDArray


def debug_print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


type Vec2d = ndarray[tuple[Literal[2]], dtype[int_]]


# % === Game Constants ===
MAX_THRUST = 100
CP_RADIUS = 600
POD_RADIUS = 400
MAP_WIDTH = 16_000
MAP_HEIGHT = 9_000

FRICTION_FACTOR = 0.85
MAX_SPEED = 567  # Maximum speed that can be reached with regular acceleration (f / (1 - f) * 100 where f is the friction and 100 the max thrust)
MINIMUM_IMPULSE = 120
BOOST_ACCELERATION = 650
SHIELD_MASS_FACTOR = 10
ROTATION_MAX_ANGLE = 18
POD_TIMEOUT = 100

NB_LAPS = int(input())
NB_CP = int(input())


# %% === User Constants ===
EPSILON = 1e-10  # For stability in operations


# %% === Functions ===
def linear_step(x: float) -> float:
    """Ramp between 0 and 1."""
    return max(0, min(1, x))


SMOOTH_POLY_3 = Polynomial((0, 0, 3, -2))


def smoothstep3(x: float) -> float:
    """3nd order polynomial smooth step between 0 and 1."""
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return SMOOTH_POLY_3(x)  # pyright: ignore[reportReturnType]


SMOOTH_POLY_5 = Polynomial((0, 0, 0, 10, -15, 6))


def smoothstep5(x: float) -> float:
    """5th order polynomial smooth step between 0 and 1."""
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return SMOOTH_POLY_5(x)  # pyright: ignore[reportReturnType]


def sigmoid_step(x: float) -> float:
    """Sigmoid smooth step between 0 and 1."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    a = exp(-1.0 / (x + EPSILON))
    b = exp(-1.0 / (1 - x + EPSILON))
    return a / (a + b)


# %% === Classes ===
class Entity(ABC):
    """A game entity with position."""

    radius: int

    def __init__(self, x: int, y: int) -> None:
        self.position: Vec2d = asarray([x, y])  # pyright:ignore[reportAttributeAccessIssue]

    @property
    def x(self) -> int:
        return self.position[0]

    @property
    def y(self) -> int:
        return self.position[1]

    def distance_from(self, other: Entity) -> floating[Any]:
        """Return the distance from the other entity."""
        return np.linalg.vector_norm(self.position - other.position, ord=2)

    def collides_with(self, other: Entity) -> bool_:
        """Return whether the entity collides with the other."""
        return self.distance_from(other) < self.radius + other.radius


class CP(Entity):
    """A checkpoint."""

    radius: int = CP_RADIUS


class Pod(Entity):
    """A racing pod."""

    radius: int = POD_RADIUS

    def __init__(self, x: int, y: int) -> None:
        super().__init__(x=x, y=y)

        self.velocity: Vec2d = asarray([0, 0])  # pyright:ignore[reportAttributeAccessIssue]

        self._cp: int = -1
        self._lap: int = 0
        self._angle: int

    @property
    def cp(self) -> int:
        return self._cp

    @cp.setter
    def cp(self, val: int) -> None:
        if val == 0 and self._cp != 0:
            self._lap += 1
        self._cp = val

    @property
    def lap(self) -> int:
        """Lap number."""
        return self._lap

    @Entity.x.setter
    def x(self, val: int) -> None:
        self.position[0] = val

    @Entity.y.setter
    def y(self, val: int) -> None:
        self.position[1] = val

    @property
    def vx(self) -> int:
        return self.velocity[0]

    @vx.setter
    def vx(self, val: int) -> None:
        self.velocity[0] = val

    @property
    def vy(self) -> int:
        return self.velocity[1]

    @vy.setter
    def vy(self, val: int) -> None:
        self.velocity[1] = val

    @property
    def angle(self) -> int:
        """Angle of the pod in degrees."""
        return self._angle

    @angle.setter
    def angle(self, x: int) -> None:
        """Invalidate derived properties caches."""
        self._angle = x
        del self.angle_rad
        del self.facing_vector

    @cached_property
    def angle_rad(self) -> float:
        """Return the angle of the pod in radians."""
        return math.radians(self.angle)

    @cached_property
    def facing_vector(self) -> Vec2d:
        """Return the normalized facing vector of the pod."""
        return np.asarray([math.cos(self.angle_rad), math.sin(self.angle_rad)])  # pyright: ignore[reportReturnType]

    def update_from_input(self, inputs: list[int]) -> None:
        self.x = inputs[0]
        self.y = inputs[1]
        self.vx = inputs[2]
        self.vy = inputs[3]
        self.angle = inputs[4]
        self.cp = inputs[5]

    def collision_impulse(self, other_pod: Pod) -> float:
        if not self.collides_with(other_pod):
            return 0

        impulse = ...
        raise NotImplementedError

        return max(MINIMUM_IMPULSE, impulse)


class SPECIAL_ACTION(StrEnum):
    """Special actions that replace thrust, such as BOOST or SHIELD."""

    BOOST = "BOOST"
    SHIELD = "SHIELD"


cp_by_id = {i + 1: CP(*[int(j) for j in input().split()]) for i in range(NB_CP)}

allied_pods = [Pod(0, 0), Pod(0, 0)]
enemy_pods = [Pod(0, 0), Pod(0, 0)]
all_pods = allied_pods + enemy_pods

boost_available = False

# game loop
while True:
    for pod in all_pods:
        pod.update_from_input([int(j) for j in input().split()])

    # You have to output the target position
    # followed by the power (0 <= thrust <= 100)
    # i.e.: "x y thrust"
    print("8000 4500 100")
    print("8000 4500 100")
