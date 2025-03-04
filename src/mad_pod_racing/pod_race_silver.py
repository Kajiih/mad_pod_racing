"""Mad Pod Race Silver."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from math import exp, sqrt
from typing import TypedDict

from numpy.polynomial import Polynomial


def debug_print(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


MAX_THRUST = 100
CP_WIDTH = 600
POD_RADIUS = 400
MAP_WIDTH = 16_000
MAP_HEIGHT = 9_000
MAP_CENTER = (MAP_WIDTH // 2, MAP_HEIGHT // 2)
TOTAL_NB_LAP = 3


class SPECIAL_ACTION(StrEnum):
    """Special actions that replace thrust, such as BOOST or SHIELD."""

    BOOST = "BOOST"
    SHIELD = "SHIELD"


# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

BOOST_MAX_ANGLE = 1
BOOST_MIN_DISTANCE = 7_000  # 6_000
SHIELD_MIN_ANGLE = 90
START_BREAK_DISTANCE = 4 * CP_WIDTH
MAX_BREAK_FACTOR = 3  # Thrust divided by this value
CHANGE_TARGET_CP_DISTANCE = (
    4 * CP_WIDTH  # 3 6
)  # Targeting the next cp if closer than this distance to the next cp
CHANGE_TARGET_CP_ANGLE = 30  # 45


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


EPSILON = 1e-10  # For stability in operations


def sigmoid_step(x: float) -> float:
    """Sigmoid smooth step between 0 and 1."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    a = exp(-1.0 / (x + EPSILON))
    b = exp(-1.0 / (1 - x + EPSILON))
    return a / (a + b)


def dist(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def vector_angle(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Return the angle in rad between 2 vectors."""
    # Compute the angle (in radians) of each vector
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])

    # Calculate the absolute difference between the two angles
    diff = abs(angle2 - angle1)

    # Normalize the difference to be between 0 and pi
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff


def are_colliding(pos1: tuple[int, int], pos2: tuple[int, int], radius: int = POD_RADIUS) -> bool:
    return dist(pos1, pos2) < 2 * radius


@dataclass
class CP:
    x: int
    y: int
    id: int

    @property
    def coords(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass
class Pod:
    x: int
    y: int
    vx: int = 0
    vy: int = 0

    @property
    def coords(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def velocity(self) -> tuple[int, int]:
        return self.vx, self.vy

    def velocity_magnitude(self) -> float:
        return sqrt(self.vx**2 + self.vy**2)

    def get_next_pos(self, thrust_x: int = 0, thrust_y: int = 0) -> tuple[int, int]:
        """Return the position of the pod at the next turn, given some applied trust (doesn't take turn limits into account)."""
        vx, vy = self.vx + thrust_x, self.vy + thrust_y

        return self.x + vx, self.y + vy

    def update_position(self, x: int, y: int) -> None:
        debug_print(f"expected position: {self.get_next_pos()}; actual position: {x, y}")
        self.vx = x - self.x
        self.x = x
        self.y = y - self.y
        self.y = y


DEFAULT_CP = CP(x=MAP_CENTER[0], y=MAP_CENTER[1], id=0)


class PodInput(TypedDict):
    x: int
    y: int
    checkpoint_x: int
    checkpoint_y: int
    checkpoint_dist: int
    checkpoint_angle: int


class OpponentInput(TypedDict):
    x: int
    y: int


class GameState:
    def __init__(self) -> None:
        self.step_nb: int = 0
        self.cumulative_cp_nb: int = 0
        self.lap_nb: int = 0

        self.previous_step_cp: CP = DEFAULT_CP
        self.current_cp: CP = DEFAULT_CP
        self.next_cp: CP = DEFAULT_CP

        self.cp_by_id: dict[int, CP] = {}
        self.cp_by_coords: dict[tuple[int, int], CP] = {}

        self.boost_available: bool = True

        self.pod: Pod
        self.opponent: Pod
        self.target_coords: tuple[int, int]

        self.current_cp_dist: int
        self.current_cp_angle: int

    @property
    def total_nb_cp(self) -> int:
        # TODO? Replace by None if not self.setup_finished else len...
        if not self.is_setup_finished():
            raise UserWarning(  # noqa: TRY003
                "Total number of checkpoint is unknown because not all checkpoints have been seen."
            )

        return len(self.cp_by_id)

    def is_setup_finished(self) -> bool:
        return self.lap_nb > 1

    def is_new_cp(self) -> bool:
        """Return whether the current step starts a new checkpoint."""
        return self.previous_step_cp.id != self.current_cp.id

    def is_new_lap(self) -> bool:
        """Return whether the current step starts a new lap."""
        return self.current_cp.id == 1 and self.is_new_cp()

    def is_last_cp(self) -> bool:
        """Return whether the current cp is the last of the race."""
        return self.lap_nb == TOTAL_NB_LAP and self.current_cp.id == self.total_nb_cp

    def step(self, pod_input: PodInput, opponent_input: OpponentInput) -> str:
        self.step_nb += 1

        # === Update CP and previous, next cp ===
        self.previous_step_cp = self.current_cp
        self.current_cp_dist = pod_input["checkpoint_dist"]
        self.current_cp_angle = pod_input["checkpoint_angle"]
        # Update pods info
        if self.step_nb == 1:
            self.pod = Pod(pod_input["x"], pod_input["y"])
            self.opponent = Pod(opponent_input["x"], opponent_input["y"])
        else:
            self.pod.update_position(pod_input["x"], pod_input["y"])
            self.opponent.update_position(opponent_input["x"], opponent_input["y"])

        # Checkpoints info
        current_cp_coords = pod_input["checkpoint_x"], pod_input["checkpoint_y"]
        current_cp = self.cp_by_coords.get(current_cp_coords)

        if current_cp is None:
            # Create new CP
            assert not self.is_setup_finished()

            current_cp = CP(
                x=current_cp_coords[0],
                y=current_cp_coords[1],
                id=len(self.cp_by_id) + 1,
            )

            self.cp_by_id[current_cp.id] = self.cp_by_coords[current_cp_coords] = current_cp

        self.current_cp = current_cp

        if self.is_setup_finished():
            self.next_cp = self.cp_by_id[(self.current_cp.id) % self.total_nb_cp + 1]

        # === Update other variables ===
        if self.is_new_cp():
            self.cumulative_cp_nb += 1
        if self.is_new_lap():
            self.lap_nb += 1

        # Compute target coordinates, action, etc
        self.target_coords = self.get_target_coords()

        # Shield if pods are gonna collide next step without thrust and the angle of collision is far from the angle of the target direction
        target_direction = (
            self.target_coords[0] - self.pod.x,
            self.target_coords[1] - self.pod.y,
        )

        pod_next_pos = self.pod.get_next_pos()
        opponent_next_pos = self.opponent.get_next_pos()
        collision_direction = (self.pod.x - self.opponent.x, self.pod.y - self.opponent.y)
        collision_direction = (
            pod_next_pos[0] - opponent_next_pos[0],
            pod_next_pos[1] - opponent_next_pos[1],
        )
        expect_collision = are_colliding(pod_next_pos, opponent_next_pos)
        debug_print(f"{expect_collision = }")
        debug_print(
            f"collision angle difference = {math.degrees(vector_angle(target_direction, collision_direction))}"
        )
        if (
            expect_collision
            and math.degrees(vector_angle(target_direction, collision_direction))
            >= SHIELD_MIN_ANGLE
        ):
            action = SPECIAL_ACTION.SHIELD
        elif self.meet_boost_condition():
            action = SPECIAL_ACTION.BOOST
            self.boost_available = False
        else:
            action = self.compute_thrust()

        self.print_state()
        # You have to output the target position
        # followed by the power (0 <= thrust <= 100)
        # i.e.: "x y thrust"
        return f"{self.target_coords[0]} {self.target_coords[1]} {action}"

    def get_target_coords(self) -> tuple[int, int]:
        # TODO: Implement smooth transitions between the next 2 checkpoints
        if (
            self.current_cp_dist < CHANGE_TARGET_CP_DISTANCE
            and abs(self.current_cp_angle) < CHANGE_TARGET_CP_ANGLE
            and not self.is_last_cp()
        ):
            return self.next_cp.coords

        return self.current_cp.coords

    def meet_boost_condition(self) -> bool:
        """
        Return whether conditions to boost are met.

        Boost has to be available, the pod has to be aligned with the next
        checkpoint and either the checkpoint far enough or being at the last cp.
        """
        return (
            self.boost_available
            and abs(self.current_cp_angle) <= BOOST_MAX_ANGLE
            and (self.current_cp_dist >= BOOST_MIN_DISTANCE or self.is_last_cp())
        )

    def compute_thrust(self) -> int:
        """Compute the thrust depending on the angle and distance to the target."""
        # Smooth acceleration
        angle_abs = abs(self.current_cp_angle)
        thrust = MAX_THRUST * sigmoid_step(1 - min(angle_abs, 90) / 90)
        # thrust = MAX_THRUST if angle_abs < 90 else 0

        # Break
        # thrust /= self.comput_break_factor()

        return int(thrust)

    # TODO? Improve and use?
    def comput_break_factor(self) -> float:
        """Compute the break factor depending on the distance to next cp."""
        return (
            -sigmoid_step((self.current_cp_dist - 1 * CP_WIDTH) / (START_BREAK_DISTANCE - CP_WIDTH))
            * (MAX_BREAK_FACTOR - 1)
            + MAX_BREAK_FACTOR
        )

    def print_state(self) -> None:
        debug_print(f"{self.cumulative_cp_nb = }")
        debug_print(f"{self.is_new_cp() = }")
        debug_print(f"{self.lap_nb = }")
        debug_print(f"{self.is_new_lap() = }")
        debug_print(f"{self.is_setup_finished() = }")
        if self.is_setup_finished():
            debug_print(f"{self.total_nb_cp = }")
        debug_print(f"{self.is_last_cp() = }")
        debug_print(f"{self.boost_available = }")
        debug_print(f"{self.previous_step_cp.id = }")
        debug_print(f"{self.current_cp.id = }")
        debug_print(f"{self.current_cp_dist = }")
        debug_print(f"{self.current_cp_angle = }")
        debug_print(f"{self.current_cp.coords = }")
        debug_print(f"{self.target_coords = }")


game = GameState()
# game loop
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, checkpoint_x, checkpoint_y, checkpoint_dist, checkpoint_angle = [
        int(i) for i in input().split()
    ]
    pod_input = PodInput(
        x=x,
        y=y,
        checkpoint_x=checkpoint_x,
        checkpoint_y=checkpoint_y,
        checkpoint_dist=checkpoint_dist,
        checkpoint_angle=checkpoint_angle,
    )
    opponent_x, opponent_y = [int(i) for i in input().split()]
    opponent_input = OpponentInput(
        x=opponent_x,
        y=opponent_y,
    )

    command = game.step(pod_input=pod_input, opponent_input=opponent_input)

    print(command)
