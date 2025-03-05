"""Attempt at implementing Centripetal Catmull-Rom splines with Scipy's CubicHermiteSpline."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

# Define the control points (x, y) as 2D vectors
POINTS = np.array([
    (7, 1.5),
    (2, 2),
    (1, 1),
    (4, 0.5),
    (3, 2.5),
    (6, 2),
    (7, 3),
])

# Extract x and y values
x = POINTS[:, 0]
y = POINTS[:, 1]


# Create a linearly spaced t variable from 0 to 1
def compute_knots(x, y, alpha=0.5):
    # Initialize t array
    t = np.zeros(len(x))

    # Compute the knots using the centripetal method
    for i in range(1, len(x)):
        # Calculate the distance between consecutive points (x_i, y_i) and (x_(i+1), y_(i+1))
        dist = np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        t[i] = t[i - 1] + dist**alpha

    return t


# Set the alpha parameter for centripetal Catmull-Rom spline
alpha = 0.5

# Compute the knot parameters t for the given control points
t = compute_knots(x, y, alpha)

# Create a linearly spaced t variable from 0 to 1
tl = np.linspace(0, 100, len(x))


# Compute the tangents using the Centripetal Catmull-Rom method for both x and y
def compute_tangents(t, x, y):
    n = len(t)
    tangents = np.zeros((n, 2))  # Two columns, one for x and one for y

    # Compute tangents for x and y values using the Catmull-Rom method
    for i in range(1, n - 1):
        tangents[i, 0] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1])  # Tangent for x
        tangents[i, 1] = (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])  # Tangent for y

    # For the first and last points, use one-sided differences
    tangents[0, 0] = (x[1] - x[0]) / (t[1] - t[0])
    tangents[0, 1] = (y[1] - y[0]) / (t[1] - t[0])
    tangents[-1, 0] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    tangents[-1, 1] = (y[-1] - y[-2]) / (t[-1] - t[-2])

    return tangents


# Compute the tangents for 2D points
tangents = compute_tangents(t, x, y)

# Create the CubicHermiteSpline object for 2D points
spline = CubicHermiteSpline(t, POINTS, tangents)

# Evaluate the spline on a finer grid of t values
t_fine = np.linspace(t[0], t[-1], 100 * len(t))
points_fine = spline(t_fine)

# Plot the results
plt.xlim(0, 8)  # x-axis limits
plt.ylim(0, 4)  # y-axis limits
plt.plot(x, y, "o", label="Control Points")
plt.plot(points_fine[:, 0], points_fine[:, 1], label="Cubic Hermite Spline")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.plot(t)
plt.plot(tl)
plt.show()
