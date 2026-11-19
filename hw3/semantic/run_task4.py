# -*- coding: utf-8 -*-
"""Task 4 runner + scorer (TA-provided, pure Python: rdflib + pySHACL).

    pixi run task4 [--group <id>]            # or: python semantic/run_task4.py

The TA's FAULTY FK + IK execution (TA.ttl) is compared with YOUR execution
(Student.ttl) to find the key differences.

Steps
  1. GROUND   run ground_execution.py -> output/Student.ttl  (your FK / IK execution)
  2. MERGE    TA.ttl + Student.ttl -> one rdflib Graph
  3. SHACL    validate it with YOUR shapes.ttl (pySHACL)       -> output/validation.ttl
  4. SPARQL   run YOUR queries/*.rq on  data + validation report
  5. SCORE    compare your flags / query results with the facts computed
              directly from the two graphs

Scoring (30 points; the written explanation of the problems you found is graded by hand):
  FK  T4-1  shape MATMUL_MISMATCH       flags exactly the TA steps whose matrix differs   4
      T4-2  shape MATMUL_INCONSISTENT   flags exactly the steps that multiply wrongly      4
      T4-3  queries/first_divergence    first wrong step per case                          3
      T4-4  queries/fk_diagnosis        wrong step + cause per case                        4
  IK  T4-5  shape IK_JOINT_LIMIT        flags exactly the TA joint states out of limits    4
      T4-6  shape IK_RESIDUAL_MISMATCH  flags exactly the TA IK results that differ        4
      T4-7  queries/ik_diagnosis        (target, cause) pairs                              7
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import rdflib
from rdflib import Namespace, RDF
from pyshacl import validate

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
HW3 = Namespace('http://taica.course/hw3/ontology#')
SH = Namespace('http://www.w3.org/ns/shacl#')
SOMA = Namespace('http://www.ease-crc.org/ont/SOMA.owl#')
TOL = 0.001
# Expected answers are only printed in TA debug mode (the grader / TA sets HW3_TA_DEBUG=1);
# students see OK / NOT OK per item and must find the differences themselves.
DEBUG = bool(os.environ.get('HW3_TA_DEBUG'))
ENTRIES = ['m{}{}'.format(i, j) for i in range(3) for j in range(4)]


def load(*paths):
    g = rdflib.Graph()
    for p in paths:
        g.parse(p, format='turtle')
    return g


# --------------------------------------------------------------------------
# facts computed directly from the data (the "answer")
# --------------------------------------------------------------------------
def matrix(g, node):
    m = np.eye(4)
    for i in range(3):
        for j in range(4):
            m[i, j] = float(g.value(node, HW3['m{}{}'.format(i, j)]))
    return m


def collect_steps(g):
    """{(producer, case, k): (step_node, A, T)}"""
    steps = {}
    for step in g.subjects(RDF.type, HW3.FKStep):
        cmp_ = g.value(step, HW3.isStepOfComputation)
        who = str(g.value(cmp_, HW3.producedBy))
        c = int(g.value(cmp_, HW3.caseIndex))
        k = int(g.value(step, HW3.stepIndex))
        steps[(who == 'TA', c, k)] = (
            step,
            matrix(g, g.value(step, HW3.hasCumulativeTransform)),
            matrix(g, g.value(step, HW3.hasLocalTransform)))
    return steps


def local(node):
    return str(node).split('#')[-1]


def ik_records(g):
    """{(is_ta, target): dict(status, residual, max_iters, node, states)}"""
    out = {}
    for ik in g.subjects(RDF.type, HW3.IKComputation):
        who = str(g.value(ik, HW3.producedBy))
        target = local(g.value(ik, HW3.solvesForTarget))
        states = []
        cfg = g.value(ik, HW3.hasJointConfiguration)
        for st in g.objects(cfg, HW3.hasJointState):
            joint = g.value(st, HW3.isStateOfJoint)
            states.append((st, int(g.value(joint, HW3.jointIndex)),
                           float(g.value(st, SOMA.hasJointPosition)),
                           float(g.value(joint, HW3.hasJointLowerLimit)),
                           float(g.value(joint, HW3.hasJointUpperLimit))))
        mi = g.value(ik, HW3.hasMaxIterations)
        out[(who == 'TA', target)] = dict(
            node=ik, status=str(g.value(ik, HW3.hasIKStatus)),
            residual=float(g.value(ik, HW3.hasResidual)),
            max_iters=int(mi) if mi is not None else None, states=states)
    return out


def expected_facts(g):
    steps = collect_steps(g)
    mismatch, inconsistent, cause = set(), set(), {}
    for (is_ta, c, k), (node, A, T) in steps.items():
        if not is_ta:
            continue
        stu = steps.get((False, c, k))
        if stu is not None and np.abs(A[:3] - stu[1][:3]).max() > TOL:
            mismatch.add((c, k))
        if k >= 1:
            P = steps[(True, c, k - 1)][1]
            if np.abs(A[:3, 3] - (P @ T)[:3, 3]).max() > TOL:
                inconsistent.add((c, k))
                if np.abs(A[:3, 3] - (T @ P)[:3, 3]).max() <= TOL:
                    cause[(c, k)] = 'OPERAND_ORDER_SWAPPED'
    first = {}
    for c, k in mismatch:
        first[c] = min(first.get(c, 99), k)
    # ---- IK ----
    iks = ik_records(g)
    ik_limit, ik_resid, ik_cause = set(), set(), set()
    for (is_ta, target), r in iks.items():
        if not is_ta:
            continue
        for st, j, v, lo, hi in r['states']:
            if v < lo or v > hi:
                ik_limit.add((target, j))
                ik_cause.add((target, 'JOINT_LIMIT_NOT_ENFORCED'))
        s_ = iks.get((False, target))
        if s_ is None:
            continue
        if abs(r['residual'] - s_['residual']) > 0.02:
            ik_resid.add(target)
        if (r['status'] != 'SOLVED' and s_['status'] == 'SOLVED'
                and r['max_iters'] is not None and s_['max_iters'] is not None
                and r['max_iters'] < s_['max_iters']):
            ik_cause.add((target, 'STOPPED_EARLY'))
    return dict(mismatch=mismatch, inconsistent=inconsistent, first=first,
                cause=cause, ik_limit=ik_limit, ik_resid=ik_resid,
                ik_cause=ik_cause)


# --------------------------------------------------------------------------
# student side: SHACL + SPARQL
# --------------------------------------------------------------------------
def run_shacl(data, shapes_path):
    shapes = load(shapes_path)
    conforms, report_graph, _ = validate(
        data, shacl_graph=shapes, inference='none', abort_on_first=False,
        meta_shacl=False, advanced=True, debug=False)
    return report_graph


def flags_by_prefix(data, report, prefix):
    """Keys of the TA nodes flagged with a message starting with prefix:
    FKStep -> (case, step); IKComputation -> target; TA IK JointState -> (target, jointIndex)."""
    out = set()
    for res in report.subjects(RDF.type, SH.ValidationResult):
        msg = str(report.value(res, SH.resultMessage))
        if not msg.startswith(prefix):
            continue
        node = report.value(res, SH.focusNode)
        if (node, RDF.type, HW3.FKStep) in data:
            cmp_ = data.value(node, HW3.isStepOfComputation)
            out.add((int(data.value(cmp_, HW3.caseIndex)),
                     int(data.value(node, HW3.stepIndex))))
        elif (node, RDF.type, HW3.IKComputation) in data:
            out.add(local(data.value(node, HW3.solvesForTarget)))
        elif (node, RDF.type, SOMA.JointState) in data:
            cfg = next(data.subjects(HW3.hasJointState, node), None)
            ik = next(data.subjects(HW3.hasJointConfiguration, cfg), None)
            joint = data.value(node, HW3.isStateOfJoint)
            if ik is not None and joint is not None:
                out.add((local(data.value(ik, HW3.solvesForTarget)),
                         int(data.value(joint, HW3.jointIndex))))
    return out


def run_query(graph, path):
    with open(path) as f:
        text = f.read()
    rows = []
    for row in graph.query(text):
        rows.append({str(k): v.toPython() for k, v in row.asdict().items()})
    return rows


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--group', default='student-group-00')
    ap.add_argument('--skip-ground', action='store_true',
                    help='reuse the existing output/Student.ttl')
    ap.add_argument('--reference', action='store_true',
                    help='debug: ground the TA reference solvers instead of your FK/IK')
    ap.add_argument('--shapes', default=os.path.join(HERE, 'shapes.ttl'))
    ap.add_argument('--queries', default=os.path.join(HERE, 'queries'))
    args = ap.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)
    student_ttl = os.path.join(OUTPUT, 'Student.ttl')
    ta_ttl = os.path.join(HERE, 'TA.ttl')

    print('== STEP 1/4 | GROUND: your FK/IK execution -> output/Student.ttl ==')
    if not args.skip_ground:
        cmd = [sys.executable, os.path.join(HERE, 'ground_execution.py'),
               '--group', args.group] + (['--reference'] if args.reference else [])
        if subprocess.call(cmd, cwd=HERE) != 0:
            print('\nERROR: grounding failed. Finish Task 1 (your_fk) and Task 2 '
                  '(your_ik) first, or run `python semantic/run_task4.py --reference` '
                  'to ground the TA reference solvers while you work on Task 4.')
            sys.exit(1)

    print('\n== STEP 2/4 | MERGE: TA.ttl + Student.ttl ==')
    data = load(ta_ttl, student_ttl)
    print('   {} triples'.format(len(data)))

    print('\n== STEP 3/4 | SHACL (pySHACL) with your shapes.ttl ==')
    report = run_shacl(data, args.shapes)
    report.serialize(os.path.join(OUTPUT, 'validation.ttl'), format='turtle')
    n_res = len(list(report.subjects(RDF.type, SH.ValidationResult)))
    print('   {} validation result(s) -> output/validation.ttl'.format(n_res))

    print('\n== STEP 4/4 | SPARQL on data + validation report, then scoring ==')
    merged = rdflib.Graph()
    for t in data:
        merged.add(t)
    for t in report:
        merged.add(t)

    exp = expected_facts(data)
    lines, total = [], 0.0

    def verdict(ok):
        return 'OK' if ok else 'NOT OK'

    def shape_item(tag, name, prefix, expected, pts):
        got = flags_by_prefix(data, report, prefix)
        sc = round(pts * jaccard(got, expected), 1)
        lines.append('[{}] {:<22} flagged {:>2}  {:<6} (+{:g})'.format(
            tag, name, len(got), verdict(got == expected), sc))
        if DEBUG and got != expected:
            lines.append('       expected {} missing {} extra {}'.format(
                len(expected), sorted(expected - got, key=str)[:6],
                sorted(got - expected, key=str)[:6]))
        return sc

    def query_rows(fname):
        try:
            return run_query(merged, os.path.join(args.queries, fname)), None
        except Exception as exc:    # unfinished / invalid query -> 0 points
            return [], '       {} error: {}'.format(fname, str(exc)[:80])

    total += shape_item('T4-1', 'MATMUL_MISMATCH', 'MATMUL_MISMATCH', exp['mismatch'], 4.0)
    total += shape_item('T4-2', 'MATMUL_INCONSISTENT', 'MATMUL_INCONSISTENT', exp['inconsistent'], 4.0)

    rows, err = query_rows('first_divergence.rq')
    if err:
        lines.append(err)
    got = {int(r['caseIndex']): int(r['firstStep']) for r in rows
           if 'caseIndex' in r and 'firstStep' in r}
    first = exp['first']
    ok = sum(1 for c, k in first.items() if got.get(c) == k)
    sc = round(max(0.0, 3.0 * (ok - len(set(got) - set(first))) / max(1, len(first))), 1)
    total += sc
    lines.append('[T4-3] first_divergence.rq      {} row(s)  {:<6} (+{:g})'
                 .format(len(got), verdict(got == first), sc))
    if DEBUG and got != first:
        lines.append('       expected {}  got {}'.format(first, got))

    rows, err = query_rows('fk_diagnosis.rq')
    if err:
        lines.append(err)
    got = {(int(r['caseIndex']), int(r['stepIndex'])): r.get('cause')
           for r in rows if 'caseIndex' in r and 'stepIndex' in r}
    cause = exp['cause']
    ok = sum(1 for key, v in cause.items() if got.get(key) == v)
    sc = round(max(0.0, 4.0 * (ok - len(set(got) - set(cause))) / max(1, len(cause))), 1)
    total += sc
    lines.append('[T4-4] fk_diagnosis.rq          {} row(s)  {:<6} (+{:g})'
                 .format(len(got), verdict(got == cause), sc))
    if DEBUG and got != cause:
        lines.append('       expected {}  got {}'.format(cause, got))

    total += shape_item('T4-5', 'IK_JOINT_LIMIT', 'IK_JOINT_LIMIT', exp['ik_limit'], 4.0)
    total += shape_item('T4-6', 'IK_RESIDUAL_MISMATCH', 'IK_RESIDUAL_MISMATCH', exp['ik_resid'], 4.0)

    rows, err = query_rows('ik_diagnosis.rq')
    if err:
        lines.append(err)
    got = {(local(r['target']), r['cause']) for r in rows if 'target' in r and 'cause' in r}
    sc = round(7.0 * jaccard(got, exp['ik_cause']), 1)
    total += sc
    lines.append('[T4-7] ik_diagnosis.rq          {} row(s)  {:<6} (+{:g})'
                 .format(len(got), verdict(got == exp['ik_cause']), sc))
    if DEBUG and got != exp['ik_cause']:
        lines.append('       missing {} extra {}'.format(
            sorted(exp['ik_cause'] - got), sorted(got - exp['ik_cause'])))
    report_lines = lines

    print('=' * 70)
    print('  Task 4 : SHACL + SPARQL diagnosis of the TA FK + IK execution')
    print('=' * 70)
    for line in report_lines:
        print('  ' + line)
    print('-' * 70)
    print('  Your Task 4 Score : {:.1f} / 30.0   (+ report explanation of the problems found, graded by hand)'
          .format(total))
    print('=' * 70)


if __name__ == '__main__':
    main()
