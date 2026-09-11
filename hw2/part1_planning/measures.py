"""ANSWER. The four geometric facts a candidate edge is described by."""
import math
from typing import List, Sequence, Tuple

import numpy as np

from testcase import HumanActivityZone, PersonalSpace, WorldPoint

Pixel = Tuple[int, int]


def obstacle_clearance(sample_points_px: Sequence[Pixel], dist_px: np.ndarray,
                       pixels_per_metre: float) -> float:
    """Metres from the tightest of these points to the nearest obstacle."""
    return float(min(dist_px[z, x] for x, z in sample_points_px)) / pixels_per_metre


def _inside_polygon(x: float, z: float, polygon: Sequence[WorldPoint]) -> bool:
    """Ray casting: an odd number of boundary crossings means inside."""
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, zi = polygon[current]
        xj, zj = polygon[previous]
        if (zi > z) != (zj > z) and x < xi + (xj - xi) * (z - zi) / (zj - zi):
            inside = not inside
        previous = current
    return inside


def enters_human_activity_zone(sample_points_world: Sequence[WorldPoint],
                            spaces: Sequence[HumanActivityZone]) -> List[str]:
    """The ids of every region one of these points falls inside."""
    return [space.id for space in spaces
            if any(_inside_polygon(x, z, space.polygon)
                   for x, z in sample_points_world)]


def enters_personal_space(sample_points_world: Sequence[WorldPoint],
                          people: Sequence[PersonalSpace]) -> List[str]:
    """The ids of every person whose egg one of these points falls inside."""
    entered = []
    for person in people:
        facing = math.radians(person.phi)
        facing_x, facing_z = math.cos(facing), math.sin(facing)
        for x, z in sample_points_world:
            offset_x = x - person.position[0]
            offset_z = z - person.position[1]
            forward = offset_x * facing_x + offset_z * facing_z
            lateral = -offset_x * facing_z + offset_z * facing_x
            reach = person.f if forward >= 0.0 else person.b
            if (forward / reach) ** 2 + (lateral / person.s) ** 2 <= 1.0:
                entered.append(person.id)
                break
    return entered


def contains_goal(goal_world: WorldPoint, space: HumanActivityZone) -> bool:
    """Does the point the run drives to lie inside this region?"""
    return _inside_polygon(goal_world[0], goal_world[1], space.polygon)
