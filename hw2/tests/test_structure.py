"""TA ONLY -- not distributed. What ontology/structure.ttl catches on a record,
and what only check_structure catches."""
from checks import Checks     # first: it puts the repo's modules on sys.path

from rdflib import RDF, XSD, Graph, Literal

import check_structure
import paths
import record_reader
from record_reader import HW2

NODES = {name: HW2[name] for name in ("node_1", "node_2", "node_3", "node_4",
                                      "node_5", "node_6")}
EDGES = {name: HW2[name] for name in ("edge_1", "edge_2", "edge_3", "edge_4")}
RUN = HW2["run_tc_01"]


def clean():
    """A small, well-formed record: a three-edge claimed path, one rejected
    candidate, one region and one person."""
    graph = Graph()
    for index, node in enumerate(NODES.values()):
        graph.add((node, RDF.type, HW2.Node))
        graph.add((node, HW2.worldX, Literal(float(index), datatype=XSD.double)))
        graph.add((node, HW2.worldZ,
                   Literal(-0.5 * float(index), datatype=XSD.double)))

    for name, (start, end, clearance) in {
            "edge_1": ("node_1", "node_2", 0.5),
            "edge_2": ("node_2", "node_3", 0.4),
            "edge_3": ("node_3", "node_4", 0.3),
            "edge_4": ("node_2", "node_5", 0.1)}.items():
        edge = EDGES[name]
        graph.add((edge, RDF.type, HW2.Edge))
        graph.add((edge, HW2.hasStartNode, NODES[start]))
        graph.add((edge, HW2.hasEndNode, NODES[end]))
        graph.add((edge, HW2.obstacleClearance,
                   Literal(clearance, datatype=XSD.double)))
        if name != "edge_4":            # edge_4 is the rejected candidate
            graph.add((edge, RDF.type, HW2.RRTTreeEdge))
            graph.add((edge, RDF.type, HW2.PathEdge))

    graph.add((HW2["zone_kitchen"], RDF.type, HW2.HumanActivityZone))
    graph.add((HW2["zone_kitchen"], HW2.containsGoal,
               Literal(True, datatype=XSD.boolean)))
    graph.add((HW2["person_1"], RDF.type, HW2.PersonalSpace))

    graph.add((RUN, RDF.type, HW2.PlanningRun))
    graph.add((RUN, HW2.goalObject, Literal("cooktop", datatype=XSD.string)))
    graph.add((RUN, HW2.hasRootNode, NODES["node_1"]))
    graph.add((RUN, HW2.hasGoalNode, NODES["node_4"]))
    return graph


def _candidate(graph, name, start, end):
    """One more well-formed hw2:Edge, so a tamper adds only what it means to."""
    edge = HW2[name]
    graph.add((edge, RDF.type, HW2.Edge))
    graph.add((edge, HW2.hasStartNode, NODES[start]))
    graph.add((edge, HW2.hasEndNode, NODES[end]))
    graph.add((edge, HW2.obstacleClearance, Literal(0.6, datatype=XSD.double)))
    return edge


def drop_coordinate(graph):
    graph.remove((NODES["node_2"], HW2.worldZ, None))


def drop_goal_object(graph):
    graph.remove((RUN, HW2.goalObject, None))


def drop_start_node(graph):
    graph.remove((RUN, HW2.hasRootNode, None))


def drop_goal_node(graph):
    graph.remove((RUN, HW2.hasGoalNode, None))


def drop_clearance(graph):
    graph.remove((EDGES["edge_2"], HW2.obstacleClearance, None))


def drop_contains_goal(graph):
    graph.remove((HW2["zone_kitchen"], HW2.containsGoal, None))


def make_coordinate_infinite(graph):
    graph.remove((NODES["node_2"], HW2.worldX, None))
    graph.add((NODES["node_2"], HW2.worldX,
               Literal(float("inf"), datatype=XSD.double)))


def make_coordinate_nan(graph):
    graph.remove((NODES["node_2"], HW2.worldX, None))
    graph.add((NODES["node_2"], HW2.worldX,
               Literal(float("nan"), datatype=XSD.double)))


def move_coordinate_off_grid(graph):
    graph.remove((NODES["node_2"], HW2.worldX, None))
    graph.add((NODES["node_2"], HW2.worldX,
               Literal(1.0e100, datatype=XSD.double)))


def unkeep_a_path_edge(graph):
    graph.remove((EDGES["edge_2"], RDF.type, HW2.RRTTreeEdge))


def tree_edge_without_base_type(graph):
    edge = _candidate(graph, "edge_7", "node_4", "node_6")
    graph.remove((edge, RDF.type, HW2.Edge))
    graph.add((edge, RDF.type, HW2.RRTTreeEdge))


def fork_the_path(graph):
    edge = _candidate(graph, "edge_5", "node_2", "node_6")
    graph.add((edge, RDF.type, HW2.RRTTreeEdge))
    graph.add((edge, RDF.type, HW2.PathEdge))


def drop_first_path_edge(graph):
    graph.remove((EDGES["edge_1"], RDF.type, HW2.PathEdge))


def gap_in_the_path(graph):
    graph.remove((EDGES["edge_2"], RDF.type, HW2.PathEdge))


def close_the_path_into_a_cycle(graph):
    edge = _candidate(graph, "edge_6", "node_4", "node_1")
    graph.add((edge, RDF.type, HW2.RRTTreeEdge))
    graph.add((edge, RDF.type, HW2.PathEdge))


def add_a_second_run(graph):
    other = HW2["run_tc_02"]
    graph.add((other, RDF.type, HW2.PlanningRun))
    graph.add((other, HW2.goalObject, Literal("door", datatype=XSD.string)))
    graph.add((other, HW2.hasRootNode, NODES["node_1"]))
    graph.add((other, HW2.hasGoalNode, NODES["node_4"]))


def broken(tamper, structure):
    """The distinct messages structure.ttl reports after one tamper."""
    graph = clean()
    tamper(graph)
    return graph, [message for message, _names
                   in record_reader.violations(graph, structure)]


def main():
    checks = Checks(f"{paths.ONTOLOGY_STRUCTURE} -- what it catches, and what "
                    "only the chain check catches")
    structure = record_reader.load_structure()

    checks.equal("a clean record raises nothing",
                 record_reader.violations(clean(), structure), [])

    for tamper, phrase in (
            (drop_coordinate, "missing its world Z coordinate"),
            (drop_goal_object, "does not say which object it was sent to"),
            (drop_start_node, "does not name the tree's root node"),
            (drop_goal_node, "does not name the goal point"),
            (drop_clearance, "does not record the clearance it measured"),
            (drop_contains_goal, "does not say whether the run's goal is inside it"),
            (make_coordinate_infinite, "outside the planning grid"),
            (make_coordinate_nan, "outside the planning grid"),
            (move_coordinate_off_grid, "outside the planning grid"),
            (unkeep_a_path_edge, "claimed as part of the path but was never kept"),
            (tree_edge_without_base_type, "must also carry its base type"),
            (fork_the_path, "The claimed path forks")):
        _graph, messages = broken(tamper, structure)
        checks.that(f"catches: {tamper.__name__.replace('_', ' ')}",
                    any(phrase in message for message in messages),
                    f"a violation saying '{phrase}'", messages or "no violation")

    # The two properties SHACL Core cannot state, which is why check_structure
    # counts the runs and walks the chain in code.
    graph, messages = broken(drop_first_path_edge, structure)
    checks.equal("SHACL alone does NOT catch a path that no longer reaches the "
                 "root", messages, [])
    checks.that("record_reader.path_chain does catch it",
                record_reader.path_chain(graph)[0] == [],
                "no chain", record_reader.path_chain(graph)[1])

    graph, messages = broken(add_a_second_run, structure)
    checks.equal("SHACL alone does NOT catch a second planning run", messages, [])
    checks.that("check_structure does catch it",
                not check_structure.check(graph, structure)[0].passed,
                "item 1 fails", "item 1 passed")

    chain, not_a_path = record_reader.path_chain(clean())
    checks.that("path_chain reads the clean path root-first",
                not not_a_path and chain == [EDGES["edge_1"], EDGES["edge_2"],
                                             EDGES["edge_3"]],
                "([edge_1, edge_2, edge_3], no complaint)",
                ([str(edge).rsplit("#", 1)[-1] for edge in chain], not_a_path))
    checks.equal("world_path turns that chain into four waypoints",
                 record_reader.world_path(clean(), chain),
                 [(0.0, -0.0), (1.0, -0.5), (2.0, -1.0), (3.0, -1.5)])

    for tamper in (gap_in_the_path, close_the_path_into_a_cycle, fork_the_path):
        graph = clean()
        tamper(graph)
        chain, why = record_reader.path_chain(graph)
        checks.that(f"path_chain catches {tamper.__name__.replace('_', ' ')}",
                    chain == [] and bool(why), "no chain, and a reason",
                    why or "accepted")

    no_claim = clean()
    for edge in list(no_claim.subjects(RDF.type, HW2.PathEdge)):
        no_claim.remove((edge, RDF.type, HW2.PathEdge))
    checks.that("an unclaimed path is reported as that, not as a broken chain",
                "no edge is typed" in record_reader.path_chain(no_claim)[1],
                "a reason saying no path is claimed",
                record_reader.path_chain(no_claim)[1])
    checks.done()


if __name__ == "__main__":
    main()
