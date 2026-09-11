"""ANSWER. The RDF record of one planning run, and the rules gate every candidate passes."""
from typing import List, Sequence, Tuple

import numpy as np
import pyshacl
from rdflib import RDF, XSD, Graph, Literal, Namespace, URIRef

import measures
from scene import MapProjection, pixel_to_world
from testcase import TestCase

Pixel = Tuple[int, int]

HW2 = Namespace("https://nycu.edu.tw/physical-ai/hw2/ontology#")
SH = Namespace("http://www.w3.org/ns/shacl#")


def clearance_threshold_m(rules_graph: Graph) -> float:
    """The lowest clearance the rules accept.

    Read out of the rules so the threshold is written in exactly one place.
    """
    for shape in rules_graph.subjects(SH.path, HW2.obstacleClearance):
        bound = rules_graph.value(shape, SH.minInclusive)
        if bound is not None:
            return float(bound)
    raise ValueError("The rules put no sh:minInclusive on hw2:obstacleClearance, "
                     "so there is no clearance for the planner to aim for.")


class RunRecorder:
    """Records every candidate of one run, and answers whether the rules accept it."""

    def __init__(self, case: TestCase, projection: MapProjection,
                 dist_px: np.ndarray, rules_graph: Graph, goal_pixel: Pixel):
        self.case = case
        self.projection = projection
        self.dist_px = dist_px
        self.rules_graph = rules_graph
        self.goal_pixel = goal_pixel
        self.pixels_per_metre = 1.0 / projection.resolution

        self.graph = Graph()
        self.graph.bind("hw2", HW2)
        self._node_iri = {}
        self._incoming_edge = {}
        self._node_count = 0
        self._edge_count = 0
        self._root_iri = None

        goal_world = pixel_to_world(goal_pixel[0], goal_pixel[1], projection)
        self._region_iri = {space.id: HW2[space.id]
                            for space in case.human_activity_zones}
        self._person_iri = {person.id: HW2[person.id] for person in case.people}
        # Every candidate is validated alongside these: a region whose goal
        # status is missing fails the rule's inner check and closes itself.
        self._goal_status = [
            (self._region_iri[space.id], HW2.containsGoal,
             Literal(measures.contains_goal(goal_world, space), datatype=XSD.boolean))
            for space in case.human_activity_zones]

        for space in case.human_activity_zones:
            self.graph.add((self._region_iri[space.id], RDF.type, HW2.HumanActivityZone))
        for triple in self._goal_status:
            self.graph.add(triple)
        for person in case.people:
            self.graph.add((self._person_iri[person.id], RDF.type, HW2.PersonalSpace))

        self._goal_iri = self._mint_node(goal_pixel)

    def _mint_node(self, pixel: Pixel) -> URIRef:
        self._node_count += 1
        iri = HW2[f"node_{self._node_count}"]
        x, z = pixel_to_world(pixel[0], pixel[1], self.projection)
        self.graph.add((iri, RDF.type, HW2.Node))
        self.graph.add((iri, HW2.worldX, Literal(x, datatype=XSD.double)))
        self.graph.add((iri, HW2.worldZ, Literal(z, datatype=XSD.double)))
        return iri

    def set_root(self, node) -> None:
        """Name the tree's root, which exists before any edge does."""
        self._root_iri = self._mint_node(node.pixel)
        self._node_iri[node] = self._root_iri

    def consider(self, start_node, end_node, sample_points_px: List[Pixel]) -> bool:
        """Record one candidate with what was measured about it, and report whether
        the rules accept it."""
        sample_points_world = [pixel_to_world(x, z, self.projection)
                               for x, z in sample_points_px]
        # The goal node was minted before the search, so the edge arriving there
        # and hw2:hasGoalNode name one individual.
        end_iri = (self._goal_iri if end_node.pixel == self.goal_pixel
                   else self._mint_node(end_node.pixel))
        self._edge_count += 1
        edge = HW2[f"edge_{self._edge_count}"]

        triples = [
            (edge, RDF.type, HW2.Edge),
            (edge, HW2.hasStartNode, self._node_iri[start_node]),
            (edge, HW2.hasEndNode, end_iri),
            (edge, HW2.obstacleClearance,
             Literal(measures.obstacle_clearance(sample_points_px, self.dist_px,
                                                 self.pixels_per_metre),
                     datatype=XSD.double)),
        ]
        # One triple per space entered; a space not named was not entered.
        triples += [(edge, HW2.entersHumanActivityZone, self._region_iri[region])
                    for region in measures.enters_human_activity_zone(
                        sample_points_world, self.case.human_activity_zones)]
        triples += [(edge, HW2.entersPersonalSpace, self._person_iri[person])
                    for person in measures.enters_personal_space(
                        sample_points_world, self.case.people)]

        candidate_graph = Graph()
        for triple in triples + self._goal_status:
            candidate_graph.add(triple)
        conforms, _, _ = pyshacl.validate(candidate_graph, shacl_graph=self.rules_graph)

        for triple in triples:
            self.graph.add(triple)
        if conforms:
            self.graph.add((edge, RDF.type, HW2.RRTTreeEdge))
            self._node_iri[end_node] = end_iri
            self._incoming_edge[end_node] = edge
        return conforms

    def claim_path(self, path_nodes: Sequence) -> None:
        """Claim the edges into these nodes, root first, as the run's final path."""
        for node in path_nodes[1:]:   # the root has no incoming edge
            self.graph.add((self._incoming_edge[node], RDF.type, HW2.PathEdge))

    def write(self, destination: str) -> None:
        """Write the record: the individuals this run created, and nothing copied
        back out of the test case."""
        run = HW2[f"run_{self.case.id}"]
        self.graph.add((run, RDF.type, HW2.PlanningRun))
        self.graph.add((run, HW2.goalObject,
                        Literal(self.case.goal_object, datatype=XSD.string)))
        self.graph.add((run, HW2.hasRootNode, self._root_iri))
        self.graph.add((run, HW2.hasGoalNode, self._goal_iri))
        self.graph.serialize(destination=destination, format="turtle")
