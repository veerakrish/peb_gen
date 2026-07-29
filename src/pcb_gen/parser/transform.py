"""2D transform for mapping a symbol's local pin coordinates to sheet-global
coordinates.

Convention (empirically verified against demos/pic_programmer wire/junction
endpoints, see project notes): mirror is applied first (mirror="x" negates Y,
mirror="y" negates X), then the instance's rotation angle is applied directly
(no sign flip) using the standard CCW rotation formula, then the instance
position is added.
"""

from __future__ import annotations

import math


def transform_point(
    local: tuple[float, float],
    angle: float,
    mirror: str | None,
    origin: tuple[float, float],
) -> tuple[float, float]:
    x, y = local
    if mirror == "x":
        y = -y
    elif mirror == "y":
        x = -x

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rx = x * cos_t - y * sin_t
    ry = x * sin_t + y * cos_t

    ox, oy = origin
    return (round(ox + rx, 3), round(oy + ry, 3))
