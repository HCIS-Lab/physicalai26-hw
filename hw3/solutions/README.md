# HW3 TA Solutions (internal use — DO NOT DISTRIBUTE to students)

Complete, runnable reference solutions for all four tasks of the student
template `hw3_template/`. All verified on this machine:
Task 1 = 30/30, Task 2 = 20/20, Task 3 = 40/40, Task 4 = 10/10.

| File | Replaces (student file) | Purpose |
|---|---|---|
| `fk_solution.py` | `fk.py` | Task 2 `your_fk` full solution (D-H chain + geometric Jacobian) |
| `ik_solution.py` | `ik.py` | Task 3 `your_ik` full solution (Damped Least Squares, 6D error) |
| `ground_solutions.py` | the 2 TODO functions of `semantic/ground_execution.py` | Task 1 grounding solution — injected via the `GROUND_SCRIPT` env var, no file overwrite |
| `shapes_solution.ttl` | `semantic/shapes.ttl` | Task 1 SHACL solution (ARM_OUT_OF_RANGE + NO_CONVERGENCE + JOINT_LIMIT shapes) |

> Grading workflow, dataset/answer-key maintenance, and student-issue
> triage live in `hw3_demo/TA_INTERNAL_GUIDE.md`.

## Usage

All commands assume the `hw3_template/` directory (or the published `hw3/`)
and the `taica-hw3` conda environment (on this machine the existing
environment is named `pdm-hw3`).

### Task 2 / Task 3 (replace the files directly)

```bash
cp ../hw3_demo/solutions/fk_solution.py fk.py
cp ../hw3_demo/solutions/ik_solution.py ik.py
python fk.py     # expected: 20.000 / 20.000
python ik.py     # expected: 40.000 / 40.000
```

### Task 4 (no downloads needed — runs on the Task 2/3 solutions)

```bash
python fsm_task.py    # expected: 10 SUCCESS episodes, 10.000 / 10.000
```

### Task 1 (no overwrite of the student python file; use the env-var override)

```bash
cp ../hw3_demo/solutions/shapes_solution.ttl semantic/shapes.ttl
GROUND_SCRIPT=$(pwd)/../hw3_demo/solutions/ground_solutions.py \
    PYTHON=~/miniconda3/envs/pdm-hw3/bin/python \
    bash semantic/run_task1.sh --group ta-full-solution
# expected: Your Task 1 Score : 30.0 / 30.0
#           [S3] TA dataset vs answer key: faulty 14/14, clean 16/16 OK (+10)
```

- `JENA_HOME` can point to an already-extracted apache-jena-4.10.0 to skip
  the download, e.g. `hw3_demo/semantic/.cache/apache-jena-4.10.0` in this repo.

## Restore after verification

Remember to restore the replaced `fk.py` / `ik.py` / `semantic/shapes.ttl`
to their TODO versions before publishing; `semantic/output/` and
`semantic/.cache/` are generated artifacts — delete them before publishing.
