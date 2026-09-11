"""TA ONLY -- not distributed. The fixed world <-> pixel projection scene.py
pins:
floor in both directions, one answer for scalars and arrays, and a grid that
covers the scan."""
import math
import random

import numpy as np

from checks import Checks
from scene import PROJECTION, load_scan, pixel_to_world, world_to_pixel


def main():
    checks = Checks("The fixed projection")
    resolution, (origin_x, origin_z) = PROJECTION.resolution, PROJECTION.origin
    size = PROJECTION.size
    checks.equal("resolution is the value scene.py pins",
                 resolution, 0.029782485701229846)
    checks.equal("origin is the value scene.py pins",
                 PROJECTION.origin, (-5.8313583476480435, -4.949014030717765))
    checks.equal("the grid is 500 x 500", size, 500)

    rng = random.Random(20260828)
    points = [(origin_x + rng.uniform(0.0, size * resolution),
               origin_z + rng.uniform(0.0, size * resolution))
              for _ in range(20000)]
    left_its_cell, not_the_floor, worst_offset = 0, 0, 0.0
    for x, z in points:
        px, pz = world_to_pixel(x, z, PROJECTION)
        back_x, back_z = pixel_to_world(px, pz, PROJECTION)
        worst_offset = max(worst_offset, x - back_x, z - back_z)
        if not (0.0 <= x - back_x < resolution and 0.0 <= z - back_z < resolution):
            left_its_cell += 1
        # The inverse must land on the flooring's own cell corner, exactly.
        if (back_x, back_z) != (origin_x + math.floor((x - origin_x) / resolution) * resolution,
                                origin_z + math.floor((z - origin_z) / resolution) * resolution):
            not_the_floor += 1
    checks.equal(f"{len(points)} random points round trip inside their own cell",
                 left_its_cell, 0)
    checks.that("no round trip moves a point as much as one pixel",
                worst_offset < resolution, f"< {resolution}", worst_offset)
    checks.equal("pixel_to_world returns exactly the corner world_to_pixel floored to",
                 not_the_floor, 0)

    at_nine_tenths = [world_to_pixel(origin_x + (cell + 0.9) * resolution,
                                     origin_z + (cell + 0.9) * resolution,
                                     PROJECTION) for cell in (0, 1, 137, 499)]
    checks.equal("a point 0.9 of a cell in stays in that cell (floor, not round)",
                 at_nine_tenths, [(0, 0), (1, 1), (137, 137), (499, 499)])
    checks.equal("a point 0.1 of a cell before the origin is pixel -1 "
                 "(floor, not truncation towards zero)",
                 world_to_pixel(origin_x - 0.1 * resolution,
                                origin_z - 0.1 * resolution, PROJECTION), (-1, -1))

    pixels = np.arange(size)
    array_x, array_z = pixel_to_world(pixels, pixels, PROJECTION)
    scalar = [pixel_to_world(int(pixel), int(pixel), PROJECTION) for pixel in pixels]
    checks.equal("pixel_to_world agrees scalar and array over all 500 pixels",
                 sum(1 for (ax, az), (sx, sz) in zip(zip(array_x, array_z), scalar)
                     if (ax, az) != (sx, sz)), 0)
    array_back = world_to_pixel(array_x, array_z, PROJECTION)
    scalar_back = [world_to_pixel(x, z, PROJECTION) for x, z in scalar]
    checks.equal("world_to_pixel agrees scalar and array over the same points",
                 sum(1 for (ax, az), (sx, sz) in
                     zip(zip(*array_back), scalar_back) if (ax, az) != (sx, sz)), 0)
    # A cell corner sits exactly on a cell boundary. Reading origin + p * r back
    # through the division can land one ulp inside the cell below, so allow a
    # one-pixel drift here rather than demanding an exact round trip.
    drift = [int(p) for p in pixels if abs(int(array_back[0][p]) - p) > 0
             or abs(int(array_back[1][p]) - p) > 0]
    checks.that("no cell corner reprojects further than one pixel away",
                all(abs(int(array_back[axis][p]) - p) <= 1
                    for p in pixels for axis in (0, 1)),
                "within 1 pixel", f"{len(drift)} corners a ulp short: {drift[:6]}")

    corners = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
    far_x, far_z = origin_x + size * resolution, origin_z + size * resolution
    world_corners = [pixel_to_world(px, pz, PROJECTION) for px, pz in corners]
    checks.equal("all four grid corners map inside the grid's world extent "
                 f"(x < {far_x:.2f}, z < {far_z:.2f})",
                 [corner for corner in world_corners
                  if not (origin_x <= corner[0] < far_x and origin_z <= corner[1] < far_z)],
                 [])

    coords, _colors = load_scan()
    scan_x, scan_z = coords[:, 0], coords[:, 2]
    extent = (round(float(scan_x.min()), 2), round(float(scan_x.max()), 2),
              round(float(scan_z.min()), 2), round(float(scan_z.max()), 2))
    checks.equal("the scan spans the extent the TA guide records",
                 extent, (-3.09, 6.24, -4.95, 9.91))
    px, pz = world_to_pixel(scan_x, scan_z, PROJECTION)
    checks.equal(f"all {len(coords)} scan points land on the grid",
                 int(np.count_nonzero((px < 0) | (px >= size) |
                                      (pz < 0) | (pz >= size))), 0)
    checks.done()


if __name__ == "__main__":
    main()
