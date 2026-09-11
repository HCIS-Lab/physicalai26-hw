"""TA ONLY -- not distributed. The truth table of ontology/rules.ttl, on the
single-candidate graphs the planner validates during the search."""
from checks import Checks     # first: it puts the repo's modules on sys.path

import pyshacl
from rdflib import RDF, XSD, Graph, Literal

import paths
from recorder import HW2, SH, clearance_threshold_m

GRADED_RULES = "grader/graded_rules.ttl"

REGION, PERSON = HW2["zone_kitchen"], HW2["person_1"]


def candidate(clearance=1.0, regions=(), people=(), goal_status=()):
    """One candidate edge as the recorder validates it: the edge's own triples,
    and the goal status of every region."""
    graph = Graph()
    edge = HW2["edge_1"]
    graph.add((edge, RDF.type, HW2.Edge))
    graph.add((edge, HW2.hasStartNode, HW2["node_1"]))
    graph.add((edge, HW2.hasEndNode, HW2["node_2"]))
    if clearance is not None:
        graph.add((edge, HW2.obstacleClearance,
                   Literal(clearance, datatype=XSD.double)))
    for region in regions:
        graph.add((edge, HW2.entersHumanActivityZone, region))
    for person in people:
        graph.add((edge, HW2.entersPersonalSpace, person))
    for region, holds_goal in goal_status:
        graph.add((region, HW2.containsGoal,
                   Literal(holds_goal, datatype=XSD.boolean)))
    return graph


def accepts(rules, **candidate_kwargs):
    conforms, _results, _text = pyshacl.validate(candidate(**candidate_kwargs),
                                                shacl_graph=rules)
    return conforms


def main():
    checks = Checks("The feasibility rules -- the truth table of "
                    f"{paths.ONTOLOGY_RULES}")
    rules = Graph().parse(paths.ONTOLOGY_RULES, format="turtle")

    declared = [float(bound) for bound, in rules.query(
        "SELECT ?bound WHERE { ?shape sh:path hw2:obstacleClearance ; "
        "sh:minInclusive ?bound }", initNs={"sh": SH, "hw2": HW2})]
    threshold = clearance_threshold_m(rules)
    checks.equal("the planner reads the one clearance bound the rules declare",
                 [threshold], declared)

    table = [
        ("clearance below the bound is refused",
         dict(clearance=threshold - 0.01), False),
        ("clearance a hair below the bound is refused",
         dict(clearance=threshold - 1e-9), False),
        ("clearance exactly at the bound is allowed",
         dict(clearance=threshold), True),
        ("clearance above the bound is allowed",
         dict(clearance=threshold + 0.5), True),
        ("an edge with no clearance recorded is refused",
         dict(clearance=None), False),
        ("an edge entering nothing is allowed", {}, True),
        ("entering a region that carries containsGoal true is allowed",
         dict(regions=[REGION], goal_status=[(REGION, True)]), True),
        ("entering a region that carries containsGoal false is refused",
         dict(regions=[REGION], goal_status=[(REGION, False)]), False),
        ("entering a region with no containsGoal triple at all is refused",
         dict(regions=[REGION]), False),
        ("entering the goal's region and a closed one is refused",
         dict(regions=[REGION, HW2["zone_living"]],
              goal_status=[(REGION, True), (HW2["zone_living"], False)]), False),
        ("entering a person is refused", dict(people=[PERSON]), False),
        ("entering a person while clearing everything else is still refused",
         dict(clearance=2.0, people=[PERSON]), False),
    ]
    for description, arguments, expected in table:
        checks.equal(description, accepts(rules, **arguments), expected)

    graded = Graph().parse(GRADED_RULES, format="turtle")
    checks.equal(f"{GRADED_RULES} grades no clearance: an edge through a wall "
                 "is not its business", accepts(graded, clearance=0.0), True)
    checks.equal(f"{GRADED_RULES} refuses a region with no containsGoal triple",
                 accepts(graded, regions=[REGION]), False)
    checks.equal(f"{GRADED_RULES} allows the region that holds the goal",
                 accepts(graded, regions=[REGION], goal_status=[(REGION, True)]), True)
    checks.equal(f"{GRADED_RULES} refuses entering a person",
                 accepts(graded, people=[PERSON]), False)
    checks.done()


if __name__ == "__main__":
    main()
