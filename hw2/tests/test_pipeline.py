"""TA ONLY -- not distributed. The reference acceptance run: the map, the ten
paths, the verifier's four items, and 96/100."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from checks import Checks, ROOT     # first: it puts the repo's modules on sys.path

import cv2
import numpy as np
from scipy import ndimage

import paths
import record_reader
from report import ITEM_NAMES
from scene import PROJECTION, load_map, world_to_pixel

CLEARANCE_M = 0.25
REACHABLE_FROM = (0.9, 4.6)


def stage(name, arguments, timeout=3600):
    """Run one program of the pipeline and return its output."""
    started = time.time()
    finished = subprocess.run([sys.executable] + arguments, cwd=ROOT, timeout=timeout,
                              capture_output=True, text=True)
    print(f"        {name}: {time.time() - started:.1f} s")
    if finished.returncode:
        raise SystemExit(f"{name} failed:\n{finished.stdout}{finished.stderr}")
    return finished.stdout


def reachable_area_m2(map_path, clearance_m, start_world):
    """The floor area a robot with this clearance can reach from start_world."""
    x_img, z_img, _colors = load_map(map_path)
    obstacles = np.zeros((PROJECTION.size, PROJECTION.size), dtype=np.uint8)
    obstacles[z_img, x_img] = 1
    distance = cv2.distanceTransform(1 - obstacles, cv2.DIST_L2, 5)
    free = distance >= clearance_m / PROJECTION.resolution
    components, _count = ndimage.label(free, structure=np.ones((3, 3)))
    start = world_to_pixel(*start_world, projection=PROJECTION)
    return (float(np.count_nonzero(components == components[start[1], start[0]]))
            * PROJECTION.resolution ** 2)


def report_items(directory, case_id):
    """One case's verifier items as {name: passed}."""
    with open(os.path.join(directory, case_id, "report.txt")) as handle:
        return {name: verdict == "PASS" for name, verdict in
                re.findall(r"^\[\d\] (.+?) \.+ (PASS|FAIL)", handle.read(), re.M)}


def main():
    checks = Checks("End to end -- the reference acceptance run")
    scratch = tempfile.mkdtemp(prefix="hw2_pipeline_")
    part1, part2 = os.path.join(scratch, "part1"), os.path.join(scratch, "part2")
    try:
        kept = stage("process-map", ["part1_planning/map_processor.py",
                                     "--part1-output", part1])
        print(f"        {kept.strip()}")
        counted = re.search(r"kept (\d+) of (\d+) points", kept)
        checks.equal("map_processor keeps the points the TA guide counts",
                     tuple(int(number) for number in counted.groups()),
                     (48204, 126700))

        area = reachable_area_m2(os.path.join(part1, paths.MAP_NAME),
                                 CLEARANCE_M, REACHABLE_FROM)
        checks.close(f"the area reachable from {REACHABLE_FROM} with "
                     f"{CLEARANCE_M} m of clearance", round(area, 2), 36.3, 0.1)

        planned = stage("part1", ["part1_planning/main.py", "--part1-output", part1])
        print("".join(f"        {line}\n" for line in planned.strip().splitlines()))

        # Asked of the records, not of what the run printed: the record is the
        # deliverable, and its claimed path is what the rest of this checks.
        cases = [f"tc_{n:02d}" for n in range(1, 11)]
        claimed = [case for case in cases
                   if record_reader.path_chain(
                       record_reader.load_record(case, part1))[0]]
        checks.equal("all ten cases find a path", claimed, cases)
        found = [(case, None) for case in claimed]

        stage("part2", ["part2_verification/main.py", "--no-draw",
                        "--part1-output", part1, "--part2-output", part2])
        verified = {case: report_items(part2, case) for case, _ in found}
        clean = [case for case, items in verified.items() if all(items.values())]
        short = {case: [name for name, passed in items.items() if not passed]
                 for case, items in verified.items() if not all(items.values())}
        checks.equal("the verifier passes all four items on eight cases",
                     len(clean), 8)
        checks.equal("the other two lose 'target reached' and nothing else",
                     sorted((case, tuple(failed)) for case, failed in short.items()),
                     sorted((case, (ITEM_NAMES[5],)) for case in short))

        stage("grade", ["grader/grade.py", "--no-draw", "--part1-output", part1,
                        "--rules", os.path.join(ROOT, paths.ONTOLOGY_RULES)])
        table = open(os.path.join(part1, "grading_results", "summary.txt")).read()
        print("".join(f"        {line}\n" for line in table.strip().splitlines()))
        marks = {row[0]: [int(number) for number in row[1:]]
                 for row in (line.split() for line in table.splitlines()
                             if line.startswith("tc_"))}
        total = sum(sum(points[:-1]) for points in marks.values())
        checks.equal("the grader scores what the TA guide says a correct "
                     "implementation scores", (total, 10 * len(marks)), (96, 100))
        lost = {case: [ITEM_NAMES[item] for item, points in enumerate(marks[case][:-1])
                       if not points] for case in marks if marks[case][-1] < 10}
        checks.equal("the only marks lost are item 6 on two cases",
                     sorted(lost.items()),
                     sorted((case, [ITEM_NAMES[5]]) for case in short))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    checks.done()


if __name__ == "__main__":
    main()
