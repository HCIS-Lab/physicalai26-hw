"""TA ONLY -- not distributed. Bad paths and Habitat failures stay case-local."""
from types import SimpleNamespace

from checks import Checks     # first: it puts the repo's modules on sys.path

import check_drive
import navigator
import record_reader
from test_structure import clean


def rejected(path):
    """The validation error for a path, or an empty string if it was accepted."""
    try:
        navigator._validate_path(path)
    except ValueError as problem:
        return str(problem)
    return ""


def main():
    checks = Checks("Malformed paths and simulator failures are contained")

    checks.that("an infinite mid-path coordinate is rejected before Habitat",
                "non-finite coordinate" in rejected([(0.0, 0.0),
                                                      (float("inf"), 1.0)]),
                "a finite-coordinate error")
    checks.that("a NaN mid-path coordinate is rejected before Habitat",
                "non-finite coordinate" in rejected([(0.0, 0.0),
                                                      (1.0, float("nan"))]),
                "a finite-coordinate error")
    too_far = (navigator.MAX_DRIVE_STEPS + 1) * navigator.MOVE_M
    checks.that("an enormous finite path is rejected instead of looping",
                "will not attempt to drive" in rejected([(0.0, 0.0),
                                                         (too_far, 0.0)]),
                "a drive-step limit error")

    graph = clean()
    chain, why = record_reader.path_chain(graph)
    checks.equal("the fixture has a valid path", why, "")
    original = navigator.drive_path
    try:
        def broken_scene(*_args, **_kwargs):
            raise AssertionError("scene file is incomplete")

        navigator.drive_path = broken_scene
        items = check_drive.check(
            graph, SimpleNamespace(goal_object="cooktop"), chain, draw_every=0)
    finally:
        navigator.drive_path = original

    checks.equal("a Habitat AssertionError becomes two failed items",
                 [(item.number, item.passed) for item in items],
                 [(5, False), (6, False)])
    checks.that("the report preserves the exception type and useful detail",
                all("AssertionError" in str(item.detail)
                    and "scene file is incomplete" in str(item.detail)
                    for item in items),
                "both item details name the contained setup failure")
    checks.done()


if __name__ == "__main__":
    main()
