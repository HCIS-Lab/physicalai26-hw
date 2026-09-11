import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paths
from testcase import select_cases

# ANSWER START
from typing import Tuple

Pixel = Tuple[int, int]

import cv2
import numpy as np
from rdflib import Graph
from scipy import ndimage

from planner import plan
from recorder import RunRecorder, clearance_threshold_m
from scene import PROJECTION, load_map, semantic_color, world_to_pixel


def choose_goal_pixel(case, x_img, z_img, colors, dist_px, start: Pixel,
                      clearance_m: float, pixels_per_metre: float) -> Pixel:
    """The pixel this run drives to: the roomiest spot within reach of the goal object."""
    of_the_object = np.all(colors == semantic_color(case.goal_object), axis=1)
    object_mask = np.zeros(dist_px.shape, dtype=np.uint8)
    object_mask[z_img[of_the_object], x_img[of_the_object]] = 1

    # The scan mislabels a few stray points, and breaks some objects into many
    # small pieces, so drop the specks without dropping the pieces.
    blobs, _ = ndimage.label(object_mask, structure=np.ones((3, 3)))
    big_enough = np.flatnonzero(np.bincount(blobs.ravel()) >= 3)
    object_mask = np.isin(blobs, big_enough[big_enough > 0]).astype(np.uint8)
    distance_to_object = cv2.distanceTransform(1 - object_mask, cv2.DIST_L2, 5)

    free = dist_px >= clearance_m * pixels_per_metre
    components, _ = ndimage.label(free, structure=np.ones((3, 3)))
    reachable = components[start[1], start[0]]
    if reachable == 0:
        raise ValueError(f"{case.id}: the start has less than {clearance_m} m of "
                         f"clearance, so the robot could never legally leave it.")

    candidates = ((components == reachable) &
                  (distance_to_object >= 0.5 * pixels_per_metre) &
                  (distance_to_object <= 1.0 * pixels_per_metre))
    if not candidates.any():
        raise ValueError(f"{case.id}: nowhere within 0.5-1.0 m of the "
                         f"{case.goal_object} is reachable from the start.")

    z, x = np.unravel_index(int(np.argmax(np.where(candidates, dist_px, -1.0))),
                            dist_px.shape)
    return int(x), int(z)


def plan_case(case, map_path: str, rng: random.Random, record_path: str) -> None:
    """Plan one case and write its record."""
    x_img, z_img, colors = load_map(map_path)
    start = world_to_pixel(case.start_world[0], case.start_world[1], PROJECTION)
    obstacles = np.zeros((PROJECTION.size, PROJECTION.size), dtype=np.uint8)
    obstacles[z_img, x_img] = 1
    # Obstacle pixels read 0.0, which is what makes a colliding edge fail the
    # clearance rule without a collision check anywhere.
    dist_px = cv2.distanceTransform(1 - obstacles, cv2.DIST_L2, 5)

    rules = Graph().parse(paths.ONTOLOGY_RULES, format="turtle")
    goal = choose_goal_pixel(case, x_img, z_img, colors, dist_px, start,
                             clearance_threshold_m(rules), 1.0 / PROJECTION.resolution)

    recorder = RunRecorder(case, PROJECTION, dist_px, rules, goal)
    path = plan(start, goal, PROJECTION.size, rng, recorder)
    recorder.write(record_path)
# ANSWER END


def main():
    parser = argparse.ArgumentParser(
        description="Plan each test case under the rules, and record every "
                    "candidate the search considered.")
    parser.add_argument("--case", default=None,
                        help="Test case id to plan (default: every case).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds each case's sampler, so a run can be repeated "
                             "(default: 0).")
    parser.add_argument("--part1-output", default=paths.PART1_OUTPUT,
                        help=f"Folder holding {paths.MAP_NAME}, and where each case's "
                             f"{paths.RECORD_NAME} is written "
                             f"(default: {paths.PART1_OUTPUT}).")
    args = parser.parse_args()

    map_path = os.path.join(args.part1_output, paths.MAP_NAME)
    if not os.path.exists(map_path):
        print(f"{map_path} not found -- run process-map first.")
        sys.exit(1)

    for case in select_cases(paths.TESTCASES, args.case):
        # Seeded per case, so planning one case alone repeats the batch exactly.
        rng = random.Random(f"{args.seed}:{case.id}")
        os.makedirs(os.path.join(args.part1_output, case.id), exist_ok=True)
        record_path = os.path.join(args.part1_output, case.id, paths.RECORD_NAME)
        if os.path.exists(record_path):
            os.remove(record_path)          
        # TODO 
        # START
        # Load the map and plan the path using RRT and write its record to record_path.
        # Draw every random choice from `rng` so the run reproduces.
        # Write the record even when the search finds no path.
        # Hint: You may need to implement the four functions to compute the geometric facts about the candidate edge and record them into an RDF graph:
        #       obstacle_clearance(), enters_human_activity_zone(), enters_personal_space(), contains_goal().
        #       And then use the shacl rules to validate this data graph.
        # END
        # ANSWER START
        plan_case(case, map_path, rng, record_path)
        # ANSWER END
    
        print(f"{case.id} -> {record_path}" if os.path.exists(record_path)
              else f"{case.id}: nothing was written to {record_path}")


if __name__ == "__main__":
    main()
