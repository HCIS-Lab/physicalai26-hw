"""TA ONLY -- not distributed. The personal-space egg, and the planner's
measurements against the grader's independent geometry on the real test cases."""
import math
import random

from checks import Checks     # first: it puts the repo's modules on sys.path

import geometry
import measures
import paths
from testcase import PersonalSpace, load_testcases

PERSON = PersonalSpace(id="p", position=(1.7, -0.4), phi=0.0,
                       f=1.2, b=0.6, s=0.4)
EXACT = 1e-6


def _at(person, phi, bearing_deg, distance):
    """The point `distance` from the person, `bearing_deg` off their facing."""
    angle = math.radians(phi + bearing_deg)
    return (person.position[0] + distance * math.cos(angle),
            person.position[1] + distance * math.sin(angle))


def _inside(point, person):
    """Whether the planner's measure puts this point inside the person's egg."""
    return measures.enters_personal_space([point], [person]) == [person.id]


def check_egg(checks):
    """The egg reaches exactly its three extents, and joins at the sides."""
    for phi in (0.0, 90.0, 180.0, 270.0, 37.5, -123.4):
        person = PERSON._replace(phi=phi)
        for bearing, reach, name in ((0.0, person.f, "f ahead"),
                                     (180.0, person.b, "b behind"),
                                     (90.0, person.s, "s to the left"),
                                     (-90.0, person.s, "s to the right")):
            inside = _inside(_at(person, phi, bearing, reach * (1 - EXACT)),
                             person)
            outside = _inside(_at(person, phi, bearing, reach * (1 + EXACT)),
                              person)
            checks.that(f"facing {phi}: the egg reaches {reach} m {name}, "
                        f"and no further",
                        inside and not outside, "inside, then outside",
                        f"{'inside' if inside else 'outside'}, then "
                        f"{'inside' if outside else 'outside'}")

    person = PERSON._replace(phi=37.5)
    for bearing in (90.0, -90.0):
        radius = person.s * (1 - EXACT)
        joins = [_inside(_at(person, person.phi, bearing + nudge, radius), person)
                 for nudge in (-1e-6, 0.0, 1e-6)]
        checks.that(f"the two halves join at {bearing:+.0f} degrees: just inside the "
                    f"boundary is inside from the front half and the back half",
                    all(joins), [True, True, True], joins)


def _segments(spaces, people, count, rng):
    """Random segments aimed near the regions and the people, so many enter one."""
    anchors = [vertex for space in spaces for vertex in space.polygon]
    anchors += [person.position for person in people]
    segments = []
    for _ in range(count):
        x, z = rng.choice(anchors)
        segments.append(((x + rng.gauss(0.0, 1.0), z + rng.gauss(0.0, 1.0)),
                         (x + rng.gauss(0.0, 1.0), z + rng.gauss(0.0, 1.0))))
    return segments


def check_agreement(checks, spaces, people):
    """The planner and the grader answer the same on random segments, outside
    the grader's margin."""
    rng = random.Random(20260828)
    segments = _segments(spaces, people, 600, rng)
    # Both must judge the same points, so that only the two implementations of
    # "is this inside" differ, not what was sampled.
    missed, spurious, in_band, entered = 0, 0, 0, 0
    for segment in segments:
        points = geometry._samples(list(segment))
        planner = (set(measures.enters_human_activity_zone(points, spaces)) |
                   set(measures.enters_personal_space(points, people)))
        touching = set().union(*geometry.path_entries(list(segment), spaces, people,
                                                      margin_m=0.0))
        graded = set().union(*geometry.path_entries(list(segment), spaces, people))
        entered += len(graded)
        missed += len(graded - planner)
        spurious += len(touching - planner) + len(planner - touching)
        in_band += len(planner - graded)

    checks.equal(f"over {len(segments)} segments, the planner reports every entry "
                 "the grader measures deeper than its margin", missed, 0)
    checks.equal("the two implementations agree exactly on bare containment "
                 "(the grader's margin set to zero)", spurious, 0)
    checks.that(f"the grader measured {entered} entries deeper than "
                f"{geometry.MARGIN_M} m", entered > 0, "> 0", entered)
    print(f"        {in_band} entries fell in the grader's "
          f"{geometry.MARGIN_M} m margin band: shaved corners, "
          f"reported by the planner and forgiven by the grader.")

    inside_disagreements, on_the_line = 0, 0
    for _ in range(4000):
        space = rng.choice(spaces)
        point = (rng.uniform(-3.1, 6.3), rng.uniform(-5.0, 9.9))
        if min(geometry._distance_to_segment(point, start, end) for start, end
               in zip(space.polygon, space.polygon[1:] + space.polygon[:1])) < 1e-9:
            on_the_line += 1
            continue
        if measures.contains_goal(point, space) != geometry.region_contains_goal(point, space):
            inside_disagreements += 1
    checks.equal("ray casting and winding agree on 4000 random points against the "
                 "real polygons", inside_disagreements, 0)


def main():
    checks = Checks("The egg, and planner geometry against the grader's")
    check_egg(checks)
    cases = load_testcases(paths.TESTCASES)
    spaces = cases[0].human_activity_zones
    # One id per person: the same "person_1" stands somewhere different in
    # every case.
    people = [person._replace(id=f"{case.id}_{person.id}")
              for case in cases for person in case.people]
    print(f"        {len(spaces)} regions and {len(people)} people from "
          f"{paths.TESTCASES}")
    check_agreement(checks, spaces, people)
    checks.done()


if __name__ == "__main__":
    main()
