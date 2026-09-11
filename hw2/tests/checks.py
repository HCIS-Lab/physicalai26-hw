"""TA ONLY -- not distributed. The check, report and exit plumbing every test
here shares; importing it also puts the repo's own modules on sys.path."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "part1_planning"),
                os.path.join(ROOT, "part2_verification"),
                os.path.join(ROOT, "grader")]


class Checks:
    """One test's checks, and the exit code they decide."""

    def __init__(self, title):
        self.title = title
        self.results = []
        print(f"{title}\n" + "=" * 72)

    def that(self, description, passed, expected="", actual=""):
        """Record one check; a failure prints what was expected and what was seen."""
        self.results.append(bool(passed))
        print(f"  {'PASS' if passed else 'FAIL'}  {description}"
              + ("" if actual == "" else f"  [{actual}]"))
        if not passed:
            print(f"          expected: {expected}\n          actual:   {actual}")
        return bool(passed)

    def equal(self, description, actual, expected):
        return self.that(description, actual == expected, expected, actual)

    def close(self, description, actual, expected, tolerance):
        return self.that(description, abs(actual - expected) <= tolerance,
                         f"{expected} +/- {tolerance}", actual)

    def done(self):
        """Print the summary line and exit non-zero if any check failed."""
        failed = self.results.count(False)
        print(f"{len(self.results)} checks, {failed} failed -- "
              f"{'FAIL' if failed else 'PASS'}: {self.title}")
        sys.exit(1 if failed else 0)
