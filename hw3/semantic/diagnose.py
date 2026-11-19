# -*- coding: utf-8 -*-
"""Tasks 1-3 -> semantic feedback (TA-provided; no student TODO).

After you run fk.py / ik.py / fsm_task.py, this module grounds a sample of the
execution records with the TA-provided grounding functions
(ground_execution.py) and validates them in pure Python with pySHACL and
exec-shapes.ttl:

    execution records --(ground_execution.py)--> output/diagnosis-data.ttl
                      --(pySHACL + exec-shapes.ttl)--> output/diagnosis-validation.ttl
                      --> per-record problem flags printed under the score

Flags: FK_INACCURATE, ARM_OUT_OF_RANGE, NO_CONVERGENCE, JOINT_LIMIT_VIOLATION
(and STRUCTURE:* problems). Purely informational and fail-soft: the numeric
scores are computed BEFORE this runs.

Record formats (see fk.py / ik.py / fsm_task.py call sites):
    fk_records: (case_id:str, joint_config[6], pose_7d[7], pose_error, jacobian_error)
    ik_records: (case_id:str, joint_config[6], status:str, residual, target_distance)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
DATA_FILE = 'diagnosis-data.ttl'
REPORT_PATH = os.path.join(OUTPUT, 'diagnosis-validation.ttl')
MAX_RECORDS = 10  # cap the grounded records so the graph stays small


def diagnose(task_label, fk_records=(), ik_records=()):
    """Ground the records, validate them with exec-shapes.ttl (pySHACL) and
    print the problem flags per record."""
    print('------------- Semantic diagnosis (Task 4 layer, TA shapes) -------------')
    fk_records, ik_records = list(fk_records), list(ik_records)
    n_total = len(fk_records) + len(ik_records)
    fk_records = fk_records[:MAX_RECORDS]
    ik_records = ik_records[:MAX_RECORDS]
    n_dropped = n_total - len(fk_records) - len(ik_records)
    if n_dropped:
        print('  WARNING: {} record(s) beyond MAX_RECORDS={} per kind were not '
              'validated'.format(n_dropped, MAX_RECORDS))
    if not fk_records and not ik_records:
        print('  nothing to diagnose')
        return

    try:
        import rdflib
        from rdflib import RDF, Namespace
        from pyshacl import validate
    except ImportError as exc:
        print('  skipped: {} (install with `pixi install`: rdflib + pyshacl)'.format(exc))
        return

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import common
    import ground_execution as ge

    # ---- grounding: TA-provided functions ----
    ge.GROUP_ID = 'diagnosis'
    lines = ge.robot_spec_to_triples(ge.get_ur5_DH_params(), common.JOINT_LIMITS)
    for case_id, q, pose_7d, pose_err, jac_err in fk_records:
        lines += ge.fk_computation_to_triples(
            case_id, list(q), list(pose_7d),
            round(float(pose_err), 6), round(float(jac_err), 6))
    for case_id, q, status, residual, distance in ik_records:
        lines += ge.ik_computation_to_triples(
            case_id, list(q), status,
            round(float(residual), 5), round(float(distance), 5))
    path = common.write_graph(DATA_FILE, lines, 'diagnosis', 'diagnose.py')

    # ---- validation: pySHACL + exec-shapes.ttl ----
    data = rdflib.Graph().parse(path, format='turtle')
    shapes = rdflib.Graph().parse(os.path.join(HERE, 'exec-shapes.ttl'),
                                  format='turtle')
    _, report, _ = validate(data, shacl_graph=shapes, inference='none',
                            advanced=True)
    report.serialize(REPORT_PATH, format='turtle')

    # ---- per-record flags (message prefix before ':') ----
    SH = Namespace('http://www.w3.org/ns/shacl#')
    labels = [r[0] for r in fk_records] + [r[0] for r in ik_records]
    flags = {label: set() for label in labels}
    for res in report.subjects(RDF.type, SH.ValidationResult):
        focus = str(report.value(res, SH.focusNode))
        prefix = str(report.value(res, SH.resultMessage)).split(':')[0].strip()
        if prefix == 'STRUCTURE':   # grounding is TA-provided: not a student problem
            continue
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
        print('  [{}] all {} sampled records conform (no problem flags)'
              .format(task_label, len(labels)))
