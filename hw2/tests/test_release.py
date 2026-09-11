"""TA ONLY -- not distributed. Derives the student skeleton and checks it the
way anything else is checked -- by running it."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from checks import Checks, ROOT     # first: it puts the repo's modules on sys.path

from rdflib import Graph

GIVEN_MODULES = ("paths.py", "scene.py", "testcase.py",
                 "part1_planning/map_processor.py", "part1_planning/main.py",
                 "part2_verification/record_reader.py",
                 "part2_verification/navigator.py", "part2_verification/render.py",
                 "part2_verification/report.py", "part2_verification/main.py")

DELETED_MODULES = ("planner", "recorder", "measures")

ENTRY_POINTS = ("part1_planning/map_processor.py", "part1_planning/main.py",
                "part2_verification/main.py")

IMPORT_EACH = """
import importlib.util, sys
sys.path[:0] = [".", "part1_planning", "part2_verification"]
for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location(path.replace("/", "."), path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        print("OK", path)
    except Exception as error:
        print("FAIL", path, repr(error))
"""


def walk(skeleton):
    """Every file in the skeleton, relative, not following the scene symlink."""
    found = []
    for directory, directories, names in os.walk(skeleton):
        directories[:] = [name for name in directories
                          if not os.path.islink(os.path.join(directory, name))]
        found += [os.path.relpath(os.path.join(directory, name), skeleton)
                  for name in names]
    return sorted(found)


def run(skeleton, arguments, timeout=180):
    return subprocess.run([sys.executable] + arguments, cwd=skeleton, timeout=timeout,
                          capture_output=True, text=True)


def main():
    checks = Checks("release.py -- the student skeleton, compiled, parsed and run")
    scratch = tempfile.mkdtemp(prefix="hw2_release_")
    skeleton = os.path.join(scratch, "skeleton")
    try:
        released = subprocess.run([sys.executable, "release.py", skeleton], cwd=ROOT,
                                  capture_output=True, text=True, timeout=300)
        checks.equal("release.py writes a skeleton and exits cleanly",
                     released.returncode, 0)
        if released.returncode:
            print(released.stdout + released.stderr)
            checks.done()

        files = walk(skeleton)
        broken = []
        for name in (name for name in files if name.endswith(".py")):
            with open(os.path.join(skeleton, name)) as handle:
                try:
                    compile(handle.read(), name, "exec")
                except SyntaxError as error:
                    broken.append(f"{name}: {error}")
        checks.equal(f"all {sum(name.endswith('.py') for name in files)} remaining "
                     ".py files compile", broken, [])

        broken = []
        for name in (name for name in files if name.endswith(".ttl")):
            try:
                Graph().parse(os.path.join(skeleton, name), format="turtle")
            except Exception as error:
                broken.append(f"{name}: {error}")
        checks.equal(f"all {sum(name.endswith('.ttl') for name in files)} remaining "
                     ".ttl files parse", broken, [])

        marked = [name for name in files
                  if b"ANSWER" in open(os.path.join(skeleton, name), "rb").read()]
        checks.equal("no file anywhere still contains the string ANSWER", marked, [])

        former_guide_name = b"STUDENT" + b"_GUIDE.tex"
        stale_guide_links = [name for name in files
                             if former_guide_name in
                             open(os.path.join(skeleton, name), "rb").read()]
        checks.equal("shipped files point to the current student-guide name",
                     stale_guide_links, [])

        checks.equal("the TA's files are gone",
                     [name for name in ("grader", "tests", "release.py",
                                        "TA_INTERNAL_GUIDE.tex")
                      if os.path.exists(os.path.join(skeleton, name))], [])
        checks.equal("part1_planning keeps only the two files the student writes in",
                     sorted(os.listdir(os.path.join(skeleton, "part1_planning"))),
                     ["main.py", "map_processor.py"])

        rules = Graph().parse(os.path.join(skeleton, "ontology", "rules.ttl"),
                              format="turtle")
        checks.equal("ontology/rules.ttl is an empty stub that still parses",
                     len(rules), 0)

        manifest = open(os.path.join(skeleton, "pixi.toml")).read()
        targets = re.findall(r'cmd = "python ([^ "]+)"', manifest)
        checks.equal("the pixi tasks left are the three the student runs, the "
                     "grade task gone with the grader", sorted(targets),
                     sorted(ENTRY_POINTS))
        checks.equal("every one of them points at a file the skeleton has",
                     [target for target in targets
                      if not os.path.exists(os.path.join(skeleton, target))], [])

        importing = re.compile(r"^\s*(?:from|import)\s+(%s)\b" % "|".join(DELETED_MODULES),
                               re.MULTILINE)
        refers = [name for name in files if name.endswith(".py")
                  and importing.search(open(os.path.join(skeleton, name)).read())]
        checks.equal(f"nothing left imports {', '.join(DELETED_MODULES)}", refers, [])

        importer = os.path.join(scratch, "import_each.py")
        open(importer, "w").write(IMPORT_EACH)
        imported = run(skeleton, [importer] + list(GIVEN_MODULES))
        failures = [line for line in imported.stdout.splitlines()
                    if not line.startswith("OK")]
        failures += imported.stderr.strip().splitlines()[-1:] if imported.returncode else []
        checks.equal(f"all {len(GIVEN_MODULES)} given modules import with the "
                     "solutions gone", failures, [])

        for entry in ENTRY_POINTS:
            helped = run(skeleton, [entry, "--help"], timeout=120)
            checks.that(f"{entry} --help returns without hanging",
                        helped.returncode == 0 and "usage:" in helped.stdout,
                        "exit 0 and a usage line",
                        f"exit {helped.returncode}: "
                        f"{(helped.stdout + helped.stderr).splitlines()[:1]}")

        output = os.path.join(scratch, "part1_output")
        os.makedirs(output)
        stub = run(skeleton, ["part1_planning/map_processor.py",
                              "--part1-output", output])
        # The skeleton runs out of the box and writes a map of the whole scan:
        # visibly not a map, which is a better first signal than a traceback.
        checks.that("the map_processor stub runs and keeps the scan untouched",
                    stub.returncode == 0 and "kept 126700 of 126700" in stub.stdout,
                    "exit 0 and 'kept 126700 of 126700'",
                    f"exit {stub.returncode}: {stub.stdout.strip().splitlines()[-1:]}")

        shutil.copy(os.path.join(ROOT, "part1_output", "map.npz"), output)
        stub = run(skeleton, ["part1_planning/main.py", "--case", "tc_01",
                              "--part1-output", output])
        checks.that("the planning stub runs and says it wrote nothing",
                    stub.returncode == 0 and "nothing was written" in stub.stdout,
                    "exit 0 and 'nothing was written'",
                    f"exit {stub.returncode}: {stub.stdout.strip().splitlines()[-1:]}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    checks.done()


if __name__ == "__main__":
    main()
