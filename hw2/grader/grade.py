"""TA ONLY -- not distributed. Marks a submitted part1_output: six items per
case, from the submitted artefacts alone.

The record is read with the verifier's reader, which is ours and given to
students intact; the geometry is recomputed here, by a different method, and is
never shared with the code being graded.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "part2_verification"),
                os.path.dirname(os.path.abspath(__file__))]

import cv2
import numpy as np
from rdflib import RDF, Graph

import paths
import scene
import testcase

import check_drive
import check_rules
import check_structure
import navigator
import record_reader
import render
import report
from report import Item

import check_spec
import geometry

GRADED_RULES = os.path.join(ROOT, "grader", "graded_rules.ttl")

SCORE = tuple(zip(report.ITEM_NAMES, (1, 1, 2, 2, 2, 2)))

# map.npz and ontology/rules.ttl are both required deliverables that no item
# scores directly: without the map, a run cannot be drawn over it; without the
# rules file, item 2 has nothing to check the record against.
MAP_PENALTY = 20
RULES_PENALTY = 20
CASE_TOTAL = sum(points for _, points in SCORE)


def grade_case(case, part1_output, structure, declared, rules_path, graded,
               draw_every):
    """The six items for one case, with the evidence behind each, and the record
    and claimed path they were derived from."""
    record_path = os.path.join(part1_output, case.id, paths.RECORD_NAME)
    if not os.path.exists(record_path):
        return _gated(Item(1, False, [f"No record submitted for this case: "
                                      f"{record_path} does not exist."])), None, None
    try:
        record = record_reader.load_record(case.id, part1_output)
    except Exception as error:
        return _gated(Item(1, False,
                           [f"{record_path} does not parse: {error}"])), None, None

    structural, chain = check_structure.check(record, structure)
    consistent = check_rules.check(record, declared, rules_path)
    if not structural.passed:
        return _scored(_gated(structural, consistent)), record, chain

    items = [structural, consistent] + check_spec.check(record, case, chain, graded) \
        + check_drive.check(record, case, chain, draw_every)
    return _scored(items), record, chain


def _scored(items):
    """The same items, each carrying what it is worth when it passed."""
    return [item._replace(points=SCORE[item.number - 1][1] if item.passed else 0)
            for item in items]


def _gated(structural, consistent=None):
    """A failed item 1, and the items it stops from being checked.

    Item 2 survives wherever there is a record to read: it asks only whether the
    kept edges satisfy the declared rules, which needs no path and no coordinates.
    """
    why = "item 1 failed, so there is no route to judge or to drive."
    second = consistent or report.skipped(2, "there is no record to read.")
    return [structural, second] + [report.skipped(number, why)
                                   for number in (3, 4, 5, 6)]


def _submitted_map(part1_output):
    """The map the submission planned on, drawn, or None if it was not handed in."""
    map_path = os.path.join(part1_output, paths.MAP_NAME)
    if not os.path.exists(map_path):
        return None
    try:
        return render.render_map(*scene.load_map(map_path), scene.PROJECTION)
    except Exception:
        # A corrupt map cannot be rendered, but records can still be scored.
        return None


def _summary(args, marks, missing_map, missing_rules):
    """The marking scheme, one row per case, and the total."""
    lines = [f"HW2 grading -- {args.part1_output}",
             f"    declared rules: {args.rules}",
             f"    test cases:     {args.testcases}",
             "",
             f"{CASE_TOTAL} points per case, each item all or nothing:"]
    lines += [f"    {title} -- {points}" for title, points in SCORE]
    lines += ["", f"{'case':<12}" + "".join(f"{number:>4}" for number in range(1, len(SCORE) + 1))
              + f"{'score':>8}"]
    for case_id, items in marks:
        lines.append(f"{case_id:<12}" + "".join(f"{item.points:>4}" for item in items)
                     + f"{sum(item.points for item in items):>8}")
    total = sum(item.points for _, items in marks for item in items)
    out_of = len(marks) * CASE_TOTAL
    lines += ["", f"{'TOTAL':<12}{'':>24}{total:>8} / {out_of}"]

    deductions = []
    if missing_rules:
        deductions.append((f"no {paths.ONTOLOGY_RULES}", RULES_PENALTY))
    if missing_map:
        deductions.append((f"no {paths.MAP_NAME}", MAP_PENALTY))
    if deductions:
        # A deduction's label ("no ontology/rules.ttl") can run past the
        # 12-wide column the case rows use, so this block sets its own width
        # instead of the case table's, and aligns only against itself.
        width = max(len(label) for label, _ in deductions) + 2
        lines += [f"{label:<{width}}{-points:>6}" for label, points in deductions]
        penalty = sum(points for _, points in deductions)
        lines.append(f"{'FINAL':<{width}}{max(total - penalty, 0):>6} / {out_of}")

    if not any(item.points for _, items in marks for item in items):
        lines += ["", "Every case scored zero. If every report says no record was "
                      "submitted,", "check that --part1-output points at the "
                      "part1_output folder inside", "the submission, not at the "
                      "submission itself."]
    if missing_rules:
        lines += ["", f"No {paths.ONTOLOGY_RULES} was handed in. It is a required "
                      "part of the submission, and item 2 cannot be scored "
                      "without it."]
    if missing_map:
        lines += ["", f"No {paths.MAP_NAME} was handed in. It is a required part of "
                      f"the submission,", "and without it the run cannot be drawn "
                      "over the map it was planned on."]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Mark one submission's planning records.")
    parser.add_argument("--fast", action="store_true",
                        help="skip more frames while driving; the actions are unchanged")
    parser.add_argument("--no-draw", action="store_true",
                        help="drive without opening a window, for marking in batch")
    parser.add_argument("--part1-output", required=True,
                        help="the submission's part1_output folder; required, because "
                             "defaulting it would quietly mark our own reference")
    parser.add_argument("--rules", required=True,
                        help="the rules file the submission declared for itself; required, "
                             "because defaulting it would mark item 2 against our own answer")
    parser.add_argument("--testcases", default=os.path.join(ROOT, paths.TESTCASES),
                        help="the test cases to grade against")
    args = parser.parse_args()

    rules_missing = not os.path.exists(args.rules)

    cases = testcase.select_cases(args.testcases, None)
    structure = record_reader.load_structure(os.path.join(ROOT, paths.ONTOLOGY_STRUCTURE))
    if rules_missing:
        # No shapes at all: check_rules reports it the same way it reports an
        # empty rules.ttl, and the summary adds the flat penalty below.
        declared = Graph()
    else:
        try:
            declared = record_reader.load_rules(args.rules)
        except Exception as error:
            sys.exit(f"{args.rules} does not parse as Turtle, so nothing can be "
                     f"marked against it: {error}")
    graded = record_reader.load_rules(GRADED_RULES)
    draw_every = 0 if args.no_draw else (
        navigator.FAST_DRAW_EVERY if args.fast else navigator.NORMAL_DRAW_EVERY)

    results = os.path.join(args.part1_output, "grading_results")
    map_image = _submitted_map(args.part1_output)
    marks = []
    for case in cases:
        directory = report.reset_dir(results, case.id)
        try:
            items, graph, chain = grade_case(case, args.part1_output, structure,
                                             declared, args.rules, graded, draw_every)
            picture = None if map_image is None or graph is None else render.try_draw_run(
                graph, map_image.copy(), case,
                record_reader.typed(graph, record_reader.HW2.RRTTreeEdge),
                set(chain or ()), scene.PROJECTION)
        except Exception as problem:
            # Preserve case independence even for an unforeseen bad record or
            # simulator failure. Normal student errors are handled more
            # specifically inside grade_case and check_drive.
            detail = str(problem).strip() or "no further details"
            failed = Item(1, False,
                          f"Grading stopped safely for this case "
                          f"({type(problem).__name__}): {detail}")
            items = _scored(_gated(failed))
            picture = None
        report.write_report(directory, "report.txt", case.id, items)
        if picture is not None:
            cv2.imwrite(os.path.join(directory, "graded.png"), picture)
        marks.append((case.id, items))

    report.write_lines(results, "summary.txt",
                       _summary(args, marks, map_image is None, rules_missing))
    print(f"Grading results written to {results}")


if __name__ == "__main__":
    main()
