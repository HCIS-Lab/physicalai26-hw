"""TA ONLY -- not distributed. Items 3 and 4: does the route really respect the
social spaces, and did the record say so honestly?

Every fact here is recomputed from the test case by grader/geometry.py, which
is written separately from the planner's measures.py. Nothing in this file
believes the record; it only compares against it.

Obstacle clearance is deliberately never recomputed. The student chooses the
threshold, and item 5 is what tests that choice.
"""
import geometry
import record_reader
import report
from record_reader import HW2
from report import Item


def _kept_but_rejected(record, graded, constraint):
    """Kept edges the instructor's rules reject for one constraint, as report lines.

    Matched on the message because pyshacl reports the property shape's own
    message; graded_rules.ttl must therefore keep the constraint's name in it.
    """
    kept = record_reader.typed(record, HW2.RRTTreeEdge)
    grouped = [(message, names)
               for message, names in record_reader.violations(record, graded,
                                                              focus=kept)
               if constraint in message.lower()]
    if not grouped:
        return []
    return ["Kept edges the instructor's rules reject:"] + \
        report.violation_lines(grouped)


def _misreported(record, chain, defined, predicate, truth):
    """Path edges whose claimed entries differ from the recomputed ones."""
    wrong = []
    for edge in chain:
        claimed = record_reader.edge_claims(record, edge, predicate) & defined
        really = truth(record_reader.edge_world(record, edge))
        if claimed != really:
            wrong.append((record_reader._name(edge), sorted(claimed), sorted(really)))
    return wrong


def _lines(wrong, noun, sample=5):
    lines = []
    for name, claimed, really in wrong[:sample]:
        lines.append(f"    {name} records {', '.join(claimed) or 'no ' + noun}, "
                     f"but really enters {', '.join(really) or 'none'}")
    if len(wrong) > sample:
        lines.append(f"    and {len(wrong) - sample} more")
    return lines


def human_activity_zone(record, case, chain, graded) -> Item:
    """Item 3."""
    goal = record_reader.goal_world(record)
    spaces = {space.id: space for space in case.human_activity_zones}
    defined = set(spaces)
    failures = []

    missing = sorted(defined - record_reader.named(record, HW2.HumanActivityZone))
    if missing:
        failures.append(f"The record never states these zones, so nothing it says "
                        f"about them can be checked: {', '.join(missing)}.")

    for zone_id, claims_goal in record_reader.region_claims(record):
        if zone_id not in spaces:
            failures.append(f"The record states a zone '{zone_id}' that this test "
                            "case does not define.")
        elif claims_goal != geometry.region_contains_goal(goal, spaces[zone_id]):
            failures.append(
                f"The record exports '{zone_id}' with containsGoal "
                f"{str(claims_goal).lower()}, but the run's declared goal "
                f"({goal[0]:.2f}, {goal[1]:.2f}) is "
                f"{'inside' if not claims_goal else 'outside'} its polygon.")

    wrong = _misreported(record, chain, defined, HW2.entersHumanActivityZone,
                         lambda edge: geometry.path_entries(
                             edge, case.human_activity_zones, ())[0])
    if wrong:
        failures.append(f"{len(wrong)} path edges record the wrong zones:")
        failures += _lines(wrong, "zone")

    entered, _people = geometry.path_entries(
        record_reader.world_path(record, chain), case.human_activity_zones, ())
    for zone_id in sorted(entered):
        if not geometry.region_contains_goal(goal, spaces[zone_id]):
            failures.append(f"The claimed path really enters '{zone_id}', deeper than "
                            f"the {geometry.MARGIN_M} m margin, and the declared goal "
                            f"({goal[0]:.2f}, {goal[1]:.2f}) is not inside it.")

    failures += _kept_but_rejected(record, graded, "human activity zone")

    return Item(3, not failures, failures or [
        "No kept edge breaks the graded human activity zone rule.",
        f"Recomputed from the test case, the claimed path enters "
        f"{', '.join(sorted(entered)) or 'no zone'}, and every recorded zone and "
        f"containsGoal agrees with the geometry."])


def personal_space(record, case, chain, graded) -> Item:
    """Item 4."""
    defined = {person.id for person in case.people}
    failures = []

    missing = sorted(defined - record_reader.named(record, HW2.PersonalSpace))
    if missing:
        failures.append(f"The record never states these people: {', '.join(missing)}.")

    wrong = _misreported(record, chain, defined, HW2.entersPersonalSpace,
                         lambda edge: geometry.path_entries(edge, (), case.people)[1])
    if wrong:
        failures.append(f"{len(wrong)} path edges record the wrong people:")
        failures += _lines(wrong, "person")

    _zones, entered = geometry.path_entries(
        record_reader.world_path(record, chain), (), case.people)
    for person_id in sorted(entered):
        failures.append(f"The claimed path really enters the personal space of "
                        f"'{person_id}', deeper than the {geometry.MARGIN_M} m margin.")

    failures += _kept_but_rejected(record, graded, "personal space")

    return Item(4, not failures, failures or [
        "No kept edge breaks the graded personal-space rule.",
        "Recomputed from the test case, the claimed path enters nobody's "
        "personal space."])


def check(record, case, chain, graded):
    """Items 3 and 4."""
    return [human_activity_zone(record, case, chain, graded),
            personal_space(record, case, chain, graded)]
