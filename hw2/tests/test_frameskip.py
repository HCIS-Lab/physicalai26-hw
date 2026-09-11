"""TA ONLY -- not distributed. Skipping frames changes nothing
but how many are painted."""
import os
import time

from checks import Checks, ROOT     # first: it puts the repo's modules on sys.path

import navigator
import paths
import record_reader

CASE = "tc_01"
DRAW_EVERY = (0, navigator.NORMAL_DRAW_EVERY, navigator.FAST_DRAW_EVERY)


def main():
    checks = Checks(f"Frame skipping does not change the drive ({CASE})")
    record = record_reader.load_record(CASE, os.path.join(ROOT, paths.PART1_OUTPUT))
    chain, not_a_path = record_reader.path_chain(record)
    checks.that(f"{CASE}'s claimed path is a chain to drive", not not_a_path,
                "one chain", not_a_path or f"{len(chain)} edges")
    world_path = record_reader.world_path(record, chain)
    goal = record_reader.goal_object(record)

    drives = {}
    for draw_every in DRAW_EVERY:
        started = time.time()
        drives[draw_every] = navigator.drive_path(world_path, goal, draw_every)
        print(f"        draw_every {draw_every:>2}: {drives[draw_every]} "
              f"in {time.time() - started:.1f} s")

    reference = drives[DRAW_EVERY[0]]
    for draw_every in DRAW_EVERY[1:]:
        drive = drives[draw_every]
        checks.equal(f"draw_every {draw_every}: the same collisions as drawing "
                     "every frame", drive.collisions, reference.collisions)
        checks.equal(f"draw_every {draw_every}: the same step count",
                     drive.steps, reference.steps)
        checks.equal(f"draw_every {draw_every}: the same final distance",
                     drive.distance_m, reference.distance_m)
    checks.equal("the drive over "
                 f"{len(world_path)} waypoints hit nothing at any setting",
                 [drive.collisions for drive in drives.values()], [0, 0, 0])
    checks.done()


if __name__ == "__main__":
    main()
