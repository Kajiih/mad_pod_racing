"""
Vectorized N-dimensional implementation of Centripetal Catmull-Rom splines.

Inspired from https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype, float64, ndarray
from numpy._typing._array_like import NDArray

QUADRUPLE_SIZE: int = 4


def num_segments(point_chain: NDArray[Any]) -> int:
    return len(point_chain) - (QUADRUPLE_SIZE - 1)


type KdimArray[S: int, T: dtype[Any]] = ndarray[tuple[S], T]


def catmull_rom_spline[S: int, T: dtype[Any]](
    P0: KdimArray[S, T],
    P1: KdimArray[S, T],
    P2: KdimArray[S, T],
    P3: KdimArray[S, T],
    resolution: int,
    alpha: float = 0.5,
) -> NDArray[float64]:
    def tj(ti: float, pi: NDArray[Any], pj: NDArray[Any]) -> float:
        l = np.linalg.norm(pj - pi, ord=2)
        return ti + l**alpha  # pyright: ignore[reportReturnType]

    t0 = 0.0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    # Vectorized calculation of parameter t for each point between t1 and t2
    t = np.linspace(t1, t2, resolution).reshape(resolution, 1)

    # Calculate the spline using vectorized formulas
    A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
    A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
    A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3

    return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


def catmull_rom_chain(points: NDArray[Any], resolution: int) -> NDArray[float64]:
    n = num_segments(points)
    total_points = n * resolution  # Total number of points across all segments
    all_splines = np.empty((total_points, points.shape[1]))  # Preallocate with the exact size
    idx = 0

    for i in range(n):
        P0, P1, P2, P3 = points[i : i + QUADRUPLE_SIZE]
        segment_points = catmull_rom_spline(P0, P1, P2, P3, resolution)
        # Fill the preallocated array with segment points
        all_splines[idx : idx + resolution] = segment_points
        idx += resolution  # Move the index to the next available slot

    return all_splines


if __name__ == "__main__":
    POINTS = np.array([(7, 1.5), (2, 2), (1, 1), (4, 0.5), (3, 2.5), (6, 2), (7, 3)])
    NUM_POINTS = 100  # Density of blue chain points between two red points

    # Generate the entire Catmull-Rom spline chain
    chain_points = catmull_rom_chain(POINTS, NUM_POINTS)

    # Plot the points
    plt.plot(chain_points[:, 0], chain_points[:, 1], c="blue")
    plt.plot(POINTS[:, 0], POINTS[:, 1], c="red", linestyle="none", marker="o")
    plt.show()
