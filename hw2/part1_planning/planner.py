"""ANSWER. RRT over the pixel grid, with the gate deciding every candidate edge."""
import math
import random
from typing import List, Optional, Tuple

Pixel = Tuple[int, int]

STEP_PX = 16
BUDGET = 6000
GOAL_SAMPLE_RATE = 0.05
GOAL_CONNECT_PX = 1.5 * STEP_PX


class Node:
    """One point of the tree, and the node it was grown from."""

    __slots__ = ("pixel", "parent")

    def __init__(self, pixel: Pixel, parent: Optional["Node"]):
        self.pixel = pixel
        self.parent = parent


def _distance(a: Pixel, b: Pixel) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _line(a: Pixel, b: Pixel) -> List[Pixel]:
    """The pixels along the straight line from a to b, both ends included."""
    steps = int(_distance(a, b))
    if steps == 0:
        return [a]
    return [(int(round(a[0] + (b[0] - a[0]) * i / steps)),
             int(round(a[1] + (b[1] - a[1]) * i / steps)))
            for i in range(steps + 1)]


def plan(start: Pixel, goal: Pixel, size: int, rng: random.Random,
         gate) -> List[Pixel]:
    """Grow a tree from start to goal, returning the pixel waypoints or [].

    Nothing here checks for collisions: an edge through a wall measures no
    clearance, and the gate is what refuses it.
    """
    root = Node(start, None)
    gate.set_root(root)
    nodes = [root]

    for _ in range(BUDGET):
        target = goal if rng.random() < GOAL_SAMPLE_RATE else (
            rng.randint(0, size - 1), rng.randint(0, size - 1))
        nearest = min(nodes, key=lambda node: _distance(node.pixel, target))
        heading = math.atan2(target[1] - nearest.pixel[1],
                             target[0] - nearest.pixel[0])
        pixel = (int(nearest.pixel[0] + STEP_PX * math.cos(heading)),
                 int(nearest.pixel[1] + STEP_PX * math.sin(heading)))
        if not (0 <= pixel[0] < size and 0 <= pixel[1] < size):
            continue

        grown = Node(pixel, nearest)
        if not gate.consider(nearest, grown, _line(nearest.pixel, pixel)):
            continue
        nodes.append(grown)

        if _distance(pixel, goal) >= GOAL_CONNECT_PX:
            continue
        arrival = Node(goal, grown)
        if not gate.consider(grown, arrival, _line(pixel, goal)):
            continue

        chain = []
        reached = arrival
        while reached is not None:
            chain.append(reached)
            reached = reached.parent
        chain.reverse()
        gate.claim_path(chain)
        return [node.pixel for node in chain]

    return []
