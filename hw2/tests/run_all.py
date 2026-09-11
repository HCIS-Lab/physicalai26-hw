"""TA ONLY -- not distributed. Runs every test, fast ones first, and reports
which passed."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Fast first, so a break shows up before anything touches the simulator.
TESTS = (("test_projection.py", False), ("test_geometry.py", False),
         ("test_rules.py", False), ("test_structure.py", False),
         ("test_robustness.py", False), ("test_release.py", False),
         ("test_frameskip.py", True),
         ("test_determinism.py", True), ("test_tamper.py", True),
         ("test_pipeline.py", True))


def main():
    parser = argparse.ArgumentParser(description="Run the HW2 tests in order.")
    parser.add_argument("--fast", action="store_true",
                        help="skip the tests that drive the simulator or re-plan")
    arguments = parser.parse_args()

    skipped = [name for name, slow in TESTS if slow and arguments.fast]
    results = []
    for name, slow in TESTS:
        if slow and arguments.fast:
            continue
        print(f"\n>>> {name}", flush=True)
        started = time.time()
        finished = subprocess.run([sys.executable, "-u", os.path.join(HERE, name)],
                                  cwd=ROOT)
        results.append((name, finished.returncode == 0, time.time() - started))

    print("\n" + "=" * 72)
    for name, passed, seconds in results:
        print(f"{name:<24}{'PASS' if passed else 'FAIL':<6}{seconds:>8.1f} s")
    failed = [name for name, passed, _seconds in results if not passed]
    print(f"{len(results) - len(failed)} of {len(results)} tests pass"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    if skipped:
        print(f"Skipped (--fast, they drive the simulator or re-plan): "
              f"{', '.join(skipped)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
