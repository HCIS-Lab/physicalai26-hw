"""TA ONLY -- not distributed. The TA guide's tamper table: each tamper must
fail the item it
names, leave the others alone, and a clean copy must still score 10/10."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from checks import Checks, ROOT     # first: it puts the repo's modules on sys.path

from rdflib import RDF, XSD, Graph, Literal

import paths
import record_reader
from record_reader import HW2
from report import ITEM_NAMES

CASE = "tc_09"
FULL_MARKS = [1, 1, 2, 2, 2, 2]

WEAKENED_RULES = """
@prefix hw2: <https://nycu.edu.tw/physical-ai/hw2/ontology#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

hw2:FeasibleEdgeShape a sh:NodeShape ;
    sh:targetClass hw2:Edge ;
    sh:property [
        sh:path hw2:obstacleClearance ; sh:datatype xsd:double ;
        sh:minCount 1 ; sh:minInclusive "0.25"^^xsd:double ;
        sh:message "Obstacle clearance below the 0.25 m this run planned under."@en ;
    ] ;
    sh:property [
        sh:path hw2:entersPersonalSpace ; sh:maxCount 0 ;
        sh:message "Edge enters a person's personal space."@en ;
    ] ;
    # States the human activity zone constraint, but permits every region. All three
    # constraints are present, so item 2 still applies; only item 3 should bite.
    sh:property [
        sh:path hw2:entersHumanActivityZone ; sh:maxCount 99 ;
        sh:message "Edge enters a human activity zone."@en ;
    ] .
"""


def submission(scratch, name):
    """A copy of the reference submission, holding one case, to tamper with."""
    destination = os.path.join(scratch, name)
    os.makedirs(os.path.join(destination, CASE))
    shutil.copy(os.path.join(ROOT, paths.PART1_OUTPUT, paths.MAP_NAME), destination)
    shutil.copy(os.path.join(ROOT, paths.PART1_OUTPUT, CASE, paths.RECORD_NAME),
                os.path.join(destination, CASE))
    return destination


def record(destination):
    """This case's record from a copied submission."""
    return Graph().parse(os.path.join(destination, CASE, paths.RECORD_NAME),
                         format="turtle")


def rewrite(graph, destination):
    """Write a tampered record back over the copy."""
    graph.serialize(destination=os.path.join(destination, CASE, paths.RECORD_NAME),
                    format="turtle")


def grade(destination, *extra):
    """Grade the submission and return this case's six item scores.

    The copy holds only this case, so the other nine gate at once on a missing
    record and nothing else is driven.
    """
    finished = subprocess.run(
        [sys.executable, "grader/grade.py", "--no-draw",
         "--part1-output", destination,
         "--rules", os.path.join(ROOT, paths.ONTOLOGY_RULES)] + list(extra),
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if finished.returncode:
        raise SystemExit(finished.stdout + finished.stderr)
    summary = open(os.path.join(destination, "grading_results",
                                "summary.txt")).read()
    row = [line.split() for line in summary.splitlines()
           if line.startswith(CASE)][0]
    return [int(points) for points in row[1:-1]]


def detail(destination):
    """The grader's written report for this case."""
    return open(os.path.join(destination, "grading_results", CASE,
                             "report.txt")).read()


def mark_a_rejected_edge_as_kept(destination, scratch):
    """A candidate the rules refused, claimed as kept: item 2 alone."""
    graph = record(destination)
    # The tightest candidate the planner threw away, and one that entered
    # nothing, so the only rule it breaks is the clearance one.
    kept = set(graph.subjects(RDF.type, HW2.RRTTreeEdge))
    refused = min((edge for edge in graph.subjects(RDF.type, HW2.Edge)
                   if edge not in kept
                   and not list(graph.objects(edge, HW2.entersHumanActivityZone))
                   and not list(graph.objects(edge, HW2.entersPersonalSpace))),
                  key=lambda edge: float(graph.value(edge, HW2.obstacleClearance)))
    graph.add((refused, RDF.type, HW2.RRTTreeEdge))
    rewrite(graph, destination)
    return ()


def delete_the_runs_goal_object(destination, scratch):
    """The run no longer says what it was sent to: item 1, and nothing below it."""
    graph = record(destination)
    graph.remove((None, HW2.goalObject, None))
    rewrite(graph, destination)
    return ()


def delete_a_nodes_coordinate(destination, scratch):
    """A node without its Z: item 1, and nothing below it."""
    graph = record(destination)
    run = graph.value(predicate=RDF.type, object=HW2.PlanningRun)
    graph.remove((graph.value(run, HW2.hasRootNode), HW2.worldZ, None))
    rewrite(graph, destination)
    return ()


def drop_the_first_path_edge(destination, scratch):
    """The chain no longer reaches the root: item 1, and the driving below it.

    Item 2 survives: it asks only about the kept edges, which are untouched.
    """
    graph = record(destination)
    run = graph.value(predicate=RDF.type, object=HW2.PlanningRun)
    root = graph.value(run, HW2.hasRootNode)
    first = [edge for edge in graph.subjects(RDF.type, HW2.PathEdge)
             if graph.value(edge, HW2.hasStartNode) == root][0]
    graph.remove((first, RDF.type, HW2.PathEdge))
    rewrite(graph, destination)
    return ()


def weaken_the_zone_rule_and_replan(destination, scratch):
    """A planner whose own rules let it cross any region: item 3."""
    repo = os.path.join(scratch, "weakened_repo")
    shutil.copytree(ROOT, repo, symlinks=True, ignore=shutil.ignore_patterns(
        ".pixi", "part1_output", "part2_output", "__pycache__", ".git", "tests"))
    rules = os.path.join(repo, paths.ONTOLOGY_RULES)
    with open(rules, "w") as handle:
        handle.write(WEAKENED_RULES)
    subprocess.run([sys.executable, "part1_planning/main.py", "--case", CASE,
                    "--part1-output", destination],
                   cwd=repo, capture_output=True, text=True, timeout=1800, check=True)
    return ("--rules", rules)


def move_a_region_onto_the_path(destination, scratch):
    """A region placed where the path actually drives: item 3."""
    graph = record(destination)
    chain, _not_a_path = record_reader.path_chain(graph)
    x, z = record_reader.world_path(graph, chain)[len(chain) // 2]
    cases = json.load(open(os.path.join(ROOT, paths.TESTCASES)))
    for case in cases["cases"]:
        if case["id"] == CASE:
            case["human_activity_zones"][2]["polygon"] = [
                [x - 0.2, z - 0.2], [x + 0.2, z - 0.2],
                [x + 0.2, z + 0.2], [x - 0.2, z + 0.2]]
    moved = os.path.join(scratch, "moved_testcases.json")
    json.dump(cases, open(moved, "w"))
    return ("--testcases", moved)


def flip_a_regions_goal_status(destination, scratch):
    """A region exported as holding the goal when it does not: item 3."""
    graph = record(destination)
    region = [region for region in graph.subjects(RDF.type, HW2.HumanActivityZone)
              if graph.value(region, HW2.containsGoal) == Literal(
                  False, datatype=XSD.boolean)][0]
    graph.remove((region, HW2.containsGoal, None))
    graph.add((region, HW2.containsGoal, Literal(True, datatype=XSD.boolean)))
    rewrite(graph, destination)
    return ()


# Each tamper, the item it must cost, and the items the table leaves alone.
TAMPERS = (
    (mark_a_rejected_edge_as_kept, 2, (1, 3, 4, 5, 6)),
    (delete_the_runs_goal_object, 1, (2,)),
    (delete_a_nodes_coordinate, 1, (2,)),
    (drop_the_first_path_edge, 1, (2,)),
    (weaken_the_zone_rule_and_replan, 3, (1, 2)),
    (move_a_region_onto_the_path, 3, (1, 2, 4, 5, 6)),
    (flip_a_regions_goal_status, 3, (1, 2, 4, 5, 6)),
)


def main():
    checks = Checks(f"The grader bites, on {CASE}")
    scratch = tempfile.mkdtemp(prefix="hw2_tamper_")
    started = time.time()
    try:
        for tamper, fails, intact in TAMPERS:
            name = tamper.__name__.replace("_", " ")
            destination = submission(scratch, tamper.__name__)
            extra = tamper(destination, scratch)
            marks = grade(destination, *extra)
            checks.equal(f"{name}: item {fails} ({ITEM_NAMES[fails - 1]}) is lost",
                         marks[fails - 1], 0)
            if intact:
                checks.equal(f"{name}: items {intact} keep their marks",
                             [marks[item - 1] for item in intact],
                             [FULL_MARKS[item - 1] for item in intact])
            else:
                checks.that(f"{name}: item 1 gates the rest, which go unchecked",
                            sum(marks) == 0 and "Not checked" in detail(destination),
                            "0/10, and a report saying 'Not checked'",
                            f"{sum(marks)}/10")
            print(f"        {name}: {marks} = {sum(marks)}/10")

        clean = submission(scratch, "clean")
        marks = grade(clean)
        checks.equal("a clean copy still scores 10/10 afterwards, so the harness "
                     "is not simply failing everything", marks, FULL_MARKS)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"        {len(TAMPERS)} tampers and one clean run in "
          f"{time.time() - started:.0f} s")
    checks.done()


if __name__ == "__main__":
    main()
