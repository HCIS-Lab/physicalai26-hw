"""TA ONLY -- not distributed. Derives the student skeleton from this
reference, so that a change here is one command away from a new skeleton."""
import os
import shutil
import sys

REFERENCE = os.path.dirname(os.path.abspath(__file__))


NOT_COPIED = (".pixi", "replica_v1", "part1_output", "part2_output", "__pycache__", ".git")

NOT_DISTRIBUTED = ("grader", "tests", "reviews", "release.py")

BLOCK_START, BLOCK_END = "# ANSWER START", "# ANSWER END"


def strip_answers(text):
    """The text with every answer block removed.

    The blank lines above and below each block collapse into a single gap.
    """
    kept, skipping, above, below = [], False, 0, None
    for line in text.splitlines():
        marker = line.strip()
        if marker == BLOCK_START:
            skipping, above = True, 0
            while above < len(kept) and not kept[-1 - above].strip():
                above += 1
        elif marker == BLOCK_END:
            skipping, below = False, 0
        elif skipping:
            continue
        elif below is not None and not marker:
            below += 1
            if below > above:
                kept.append(line)
        else:
            below = None
            kept.append(line)
    return "\n".join(kept) + "\n"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python release.py <destination>")
    destination = os.path.abspath(sys.argv[1])
    if os.path.exists(destination):
        sys.exit(f"{destination} already exists; remove it or name somewhere else.")

    shutil.copytree(REFERENCE, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*NOT_COPIED))

    removed = []
    for name in NOT_DISTRIBUTED:
        path = os.path.join(destination, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(name + "/")
        elif os.path.exists(path):
            os.remove(path)
            removed.append(name)

    
    manifest = os.path.join(destination, "pixi.toml")
    with open(manifest) as handle:
        tasks = handle.readlines()
    with open(manifest, "w") as handle:
        handle.writelines(line for line in tasks
                          if not line.startswith("grade = ")
                          and not line.startswith("# TA only"))

    stripped = []
    for directory, directories, names in os.walk(destination):
        directories.sort()
        for name in sorted(names):
            if not name.endswith((".py", ".ttl")):
                continue
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, destination)
            with open(path) as handle:
                text = handle.read()
            if text.startswith(('"""ANSWER', "# ANSWER")):
                os.remove(path)
                removed.append(relative)
            elif BLOCK_START in text:
                with open(path, "w") as handle:
                    handle.write(strip_answers(text))
                stripped.append(relative)

    print("Removed:")
    print("".join(f"    {name}\n" for name in removed), end="")
    print("Answer blocks stripped from:")
    print("".join(f"    {name}\n" for name in stripped), end="")
    print("Dropped the grade task from pixi.toml.")
    print(f"Skeleton written to {destination}")


if __name__ == "__main__":
    main()
