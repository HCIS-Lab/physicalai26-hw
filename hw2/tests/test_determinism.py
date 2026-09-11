"""TA ONLY -- not distributed. The same seed gives the same record, whether a
case is planned alone or as part of the batch."""
import os
import shutil
import subprocess
import sys
import tempfile
import time

from checks import Checks, ROOT     # first: it puts the repo's modules on sys.path

from rdflib import Graph
from rdflib.compare import isomorphic

import paths

CASE = "tc_01"


def plan(destination, *arguments):
    """Plan into a scratch folder holding a copy of the reference map."""
    os.makedirs(destination, exist_ok=True)
    shutil.copy(os.path.join(ROOT, paths.PART1_OUTPUT, paths.MAP_NAME), destination)
    started = time.time()
    finished = subprocess.run(
        [sys.executable, "part1_planning/main.py", "--part1-output", destination]
        + list(arguments), cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if finished.returncode:
        raise SystemExit(finished.stdout + finished.stderr)
    print(f"        {' '.join(arguments) or 'the whole batch'}: "
          f"{time.time() - started:.1f} s")
    return Graph().parse(os.path.join(destination, CASE, paths.RECORD_NAME),
                         format="turtle")


def main():
    checks = Checks("The planner is reproducible -- same seed, same tree")
    scratch = tempfile.mkdtemp(prefix="hw2_determinism_")
    try:
        first = plan(os.path.join(scratch, "first"), "--case", CASE, "--seed", "0")
        again = plan(os.path.join(scratch, "again"), "--case", CASE, "--seed", "0")
        batch = plan(os.path.join(scratch, "batch"), "--seed", "0")
        other = plan(os.path.join(scratch, "other"), "--case", CASE, "--seed", "7")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    checks.equal(f"{CASE} twice at seed 0: the same number of triples",
                 len(again), len(first))
    checks.that(f"{CASE} twice at seed 0: isomorphic records",
                isomorphic(first, again), True, isomorphic(first, again))
    checks.equal("planned alone or in the batch: the same number of triples",
                 len(batch), len(first))
    checks.that("planned alone or in the batch: isomorphic records",
                isomorphic(first, batch), True, isomorphic(first, batch))
    checks.that("a different seed gives a different record, so the seed is doing "
                "something", not isomorphic(first, other),
                "not isomorphic", f"{len(first)} triples against {len(other)}")
    checks.done()


if __name__ == "__main__":
    main()
