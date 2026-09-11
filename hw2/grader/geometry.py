"""TA ONLY -- not distributed. Which social spaces a path really enters,
recomputed from the test case.

This must not be merged with the planner's own measurements: it answers the
same question by a different method -- how deep a segment reaches inside a
region, rather than whether sampled points fall in it -- so that the two
agreeing is evidence rather than a shared bug.
"""
import math

# Below this depth a path shaved a corner; above it, it drove through.
MARGIN_M = 0.05

SAMPLE_STEP_M = 0.01


def path_entries(world_path, spaces, people, margin_m=MARGIN_M):
    """The human activity zones and the people this path penetrates by more than
    the margin, as two sets of ids."""
    points = _samples(world_path)
    regions = {space.id for space in spaces
               if max((_polygon_penetration(point, space.polygon) for point in points),
                      default=0.0) > margin_m}
    persons = {person.id for person in people
               if max((_egg_penetration(point, person) for point in points),
                      default=0.0) > margin_m}
    return regions, persons


def region_contains_goal(goal, space):
    """Whether the goal point lies inside the region's polygon."""
    return _inside(goal, space.polygon)


def _samples(world_path):
    """The path resampled to one point per centimetre, corners included."""
    points = list(world_path[:1])
    for (x0, z0), (x1, z1) in zip(world_path, world_path[1:]):
        steps = max(int(math.hypot(x1 - x0, z1 - z0) / SAMPLE_STEP_M), 1)
        points += [(x0 + (x1 - x0) * step / steps, z0 + (z1 - z0) * step / steps)
                   for step in range(1, steps + 1)]
    return points


def _polygon_penetration(point, polygon):
    """How far inside the polygon the point lies, or 0.0 if it is outside."""
    if not _inside(point, polygon):
        return 0.0
    return min(_distance_to_segment(point, start, end)
               for start, end in zip(polygon, polygon[1:] + polygon[:1]))


def _egg_penetration(point, person):
    """How far inside the egg the point lies, along the ray from the person
    through it, or 0.0 if it is outside."""
    facing = math.radians(person.phi)
    offset_x = point[0] - person.position[0]
    offset_z = point[1] - person.position[1]
    forward = offset_x * math.cos(facing) + offset_z * math.sin(facing)
    lateral = -offset_x * math.sin(facing) + offset_z * math.cos(facing)

    radius = math.hypot(forward, lateral)
    if radius == 0.0:
        return min(person.f, person.b, person.s)
    reach = person.f if forward >= 0.0 else person.b
    boundary = radius / math.hypot(forward / reach, lateral / person.s)
    return max(boundary - radius, 0.0)


def _inside(point, polygon):
    """Whether the point is inside the polygon, by winding number."""
    x, z = point
    winding = 0
    for (x0, z0), (x1, z1) in zip(polygon, polygon[1:] + polygon[:1]):
        side = (x1 - x0) * (z - z0) - (x - x0) * (z1 - z0)
        if z0 <= z < z1 and side > 0:
            winding += 1
        elif z1 <= z < z0 and side < 0:
            winding -= 1
    return winding != 0


def _distance_to_segment(point, start, end):
    """The shortest distance from the point to the segment."""
    run_x, run_z = end[0] - start[0], end[1] - start[1]
    length_squared = run_x * run_x + run_z * run_z
    along = 0.0 if length_squared == 0.0 else min(1.0, max(0.0, (
        (point[0] - start[0]) * run_x + (point[1] - start[1]) * run_z) / length_squared))
    return math.hypot(point[0] - start[0] - along * run_x,
                      point[1] - start[1] - along * run_z)
