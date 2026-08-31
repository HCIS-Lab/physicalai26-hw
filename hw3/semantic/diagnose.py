# -*- coding: utf-8 -*-
"""Tasks 2-4 -> Task 1 bridge (TA-provided; no student TODO).

While you implement and run Tasks 2-4, this module feeds the numeric
execution records back through the semantic layer YOU built in Task 1:

    execution records --(your ground_execution.py functions)--> diagnosis-data.ttl
                      --(Jena shacl + your shapes.ttl)-------> diagnosis-validation.ttl
                      --> per-record problem flags printed under the score

So the classes, forms, and thresholds you defined up front in Task 1 give
immediate feedback on every FK/IK/FSM run: FK_INACCURATE, ARM_OUT_OF_RANGE,
NO_CONVERGENCE, and JOINT_LIMIT_VIOLATION are discovered by SHACL, not by
ad-hoc print statements.

Purely informational and fail-soft: numeric scores are computed BEFORE this
runs, and it degrades to a one-line notice when Task 1 is not finished yet
(grounding TODOs unimplemented) or the Jena toolchain is unavailable.

Record formats (see fk.py / ik.py / fsm_task.py call sites):
    fk_records: (case_id:str, joint_config[6], pose_7d[7], pose_error, jacobian_error)
    ik_records: (case_id:str, joint_config[6], status:str, residual, target_distance)
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
DATA_FILE = 'diagnosis-data.ttl'
REPORT_PATH = os.path.join(OUTPUT, 'diagnosis-validation.ttl')

RESULT_SPLIT = re.compile(r'(?:\ba|rdf:type)\s+sh:ValidationResult')
MAX_RECORDS = 10  # cap the grounded records so the graph stays small


def _jena_shacl():
    home = os.environ.get('JENA_HOME') or os.path.join(
        HERE, '.cache', 'apache-jena-4.10.0')
    path = os.path.join(home, 'bin', 'shacl')
    return path if os.path.exists(path) else None


def _parse_report(txt):
    results = []
    for chunk in RESULT_SPLIT.split(txt)[1:]:
        focus = re.search(r'sh:focusNode\s+([^\s;]+)', chunk)
        msg = re.search(r'sh:resultMessage\s+"([^"]*)"', chunk)
        results.append((focus.group(1) if focus else '',
                        msg.group(1) if msg else ''))
    return results


def diagnose(task_label, fk_records=(), ik_records=()):
    """Ground the records with the student's Task 1 functions, validate them
    with the student's shapes.ttl, and print the problem flags per record."""
    print('------------- Semantic diagnosis (your Task 1 layer) -------------')
    fk_records = list(fk_records)[:MAX_RECORDS]
    ik_records = list(ik_records)[:MAX_RECORDS]
    if not fk_records and not ik_records:
        print('  nothing to diagnose')
        return

    shacl = _jena_shacl()
    if shacl is None:
        print('  skipped: Apache Jena not found — run Task 1 once first'
              ' (bash semantic/run_task1.sh) or set JENA_HOME')
        return

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import common
    import ground_execution as ge

    # ---- grounding: REUSE the student's own Task 1 functions ----
    try:
        lines = ge.robot_spec_to_triples(ge.get_ur5_DH_params(),
                                         common.JOINT_LIMITS)
        for case_id, q, pose_7d, pose_err, jac_err in fk_records:
            lines += ge.fk_computation_to_triples(
                case_id, list(q), list(pose_7d),
                round(float(pose_err), 6), round(float(jac_err), 6))
        for case_id, q, status, residual, distance in ik_records:
            lines += ge.ik_computation_to_triples(
                case_id, list(q), status,
                round(float(residual), 5), round(float(distance), 5))
    except NotImplementedError:
        print('  skipped: Task 1 grounding TODOs are not implemented yet')
        return
    common.write_graph(DATA_FILE, lines, 'diagnosis', 'diagnose.py')

    # ---- validation: the student's own shapes.ttl ----
    proc = subprocess.run(
        [shacl, 'validate', '--shapes', os.path.join(HERE, 'shapes.ttl'),
         '--data', os.path.join(OUTPUT, DATA_FILE)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report = proc.stdout.decode('utf-8', 'replace')
    if not report.strip():
        print('  skipped: shacl CLI produced no report ({})'.format(
            proc.stderr.decode('utf-8', 'replace').strip()[:100]))
        return
    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    # ---- per-record flags (message prefix before ':') ----
    labels = [r[0] for r in fk_records] + [r[0] for r in ik_records]
    flags = {label: set() for label in labels}
    for focus, msg in _parse_report(report):
        prefix = msg.split(':')[0].strip()
        for label in labels:
            if label in focus:
                flags[label].add(prefix)
    flagged = [(label, sorted(flags[label])) for label in labels if flags[label]]
    if flagged:
        for label, fl in flagged:
            print('  [{}] {} -> {}'.format(task_label, label, ', '.join(fl)))
        print('  ({}/{} records flagged; full report: semantic/output/{})'
              .format(len(flagged), len(labels), 'diagnosis-validation.ttl'))
    else:
        print('  [{}] all {} sampled records conform to your shapes '
              '(no problem flags)'.format(task_label, len(labels)))
