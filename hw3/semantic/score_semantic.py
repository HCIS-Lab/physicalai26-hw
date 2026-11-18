# -*- coding: utf-8 -*-
"""Task 1 scoring script (TA-provided; grading uses the same code).

Total 30 points — grounding + SHACL:

    S1 REUSE grounding structure ............... 12 pts
       STRUCTURE:* shapes from ta-shapes-full.ttl validate output/data.ttl
       Zero violations earns full credit (each violation costs 2 points).
    S2 Problem detection on YOUR data .......... 8 pts
       TA problem shapes validate your data.ttl:
       - target_mid / target_far are flagged ARM_OUT_OF_RANGE (+2 each)
       - target_near has no problem flag (+2)
       - no JOINT_LIMIT_VIOLATION (+2)
    S3 YOUR shapes.ttl vs TA dataset + answer key  10 pts
       Your shapes.ttl validates the TA-provided 30-record execution dataset
       (ta-faulty-execution.ttl). Each record's flag set must match the
       answer key (ta-answer-key.json) exactly: no false positives and no
       false negatives.
       Score = 8 × (correct faulty cases / all faulty cases)
             + 2 × (correct clean cases / all clean cases).

Invoked automatically by STEP 5 of run_task1.sh.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')

RESULT_SPLIT = re.compile(r'(?:\ba|rdf:type)\s+sh:ValidationResult')


def parse_report(name):
    """Parse a SHACL validation report (Turtle) into [(focus, message)]."""
    path = os.path.join(OUTPUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        txt = f.read()
    if 'sh:conforms' not in txt:   # empty / unparsable report (e.g. Turtle syntax error)
        return None
    results = []
    for chunk in RESULT_SPLIT.split(txt)[1:]:
        focus = re.search(r'sh:focusNode\s+([^\s;]+)', chunk)
        msg = re.search(r'sh:resultMessage\s+"([^"]*)"', chunk)
        results.append((focus.group(1) if focus else '',
                        msg.group(1) if msg else ''))
    return results


def missing_grounding():
    """Existence checks on output/data.ttl: 1 cora:Robot, 6 RevoluteJoints, 3 IKComputations."""
    path = os.path.join(OUTPUT, 'data.ttl')
    try:
        with open(path) as f:
            txt = f.read()
    except OSError:
        return ['data.ttl missing']
    checks = [
        ('exactly one cora:Robot', len(re.findall(r'\ba\s+cora:Robot\b', txt)) == 1),
        ('6 soma:RevoluteJoint', len(re.findall(r'\ba\s+soma:RevoluteJoint\b', txt)) == 6),
        ('3 hw3:IKComputation', len(re.findall(r'\ba\s+hw3:IKComputation\b', txt)) == 3),
    ]
    return ['missing ' + name for name, ok in checks if not ok]


def flagged(results, focus_key, msg_prefix=None):
    for focus, msg in results:
        if focus_key in focus and (msg_prefix is None
                                   or msg.startswith(msg_prefix)):
            return True
    return False


def student_summary():
    """Student package (no TA shapes / answer key): show what YOUR shapes found."""
    print('=' * 70)
    print('  Task 1 : your own validation results (final score = TA grader)')
    print('=' * 70)
    missing = missing_grounding()
    print('  grounding nodes: {}'.format('OK' if not missing else ', '.join(missing)))
    for name in ('validation.ttl', 'probe-validation.ttl'):
        res = parse_report(name)
        if res is None:
            print('  {}: missing or invalid report'.format(name))
            continue
        print('  {}: {} result(s)'.format(name, len(res)))
        for focus, msg in res[:30]:
            print('     - {} : {}'.format(focus, msg[:70]))
    print('=' * 70)


def main():
    if not os.path.exists(os.path.join(HERE, 'ta-shapes-full.ttl')) \
            or not os.path.exists(os.path.join(HERE, 'ta-answer-key.json')):
        student_summary()
        sys.exit(0)
    total = 0.0
    report = []

    # ---------------- S1: grounding structure (12) ----------------
    ta = parse_report('ta-validation.ttl')
    ta_valid = ta is not None
    if ta is None:
        report.append('[S1] ta-validation.ttl missing or invalid ... FAIL (+0)')
        ta = []
    else:
        structural = [(f, m) for f, m in ta if m.startswith('STRUCTURE')]
        s1 = max(0.0, 12.0 - 2.0 * len(structural))
        # The shapes only check nodes that exist, so also require that the
        # expected grounding nodes exist (otherwise `return []` scores full S1).
        missing = missing_grounding()
        s1 = max(0.0, s1 - 4.0 * len(missing))
        structural = structural + [('(data.ttl)', 'STRUCTURE: ' + m) for m in missing]
        total += s1
        if structural:
            report.append('[S1] grounding structure .............. PARTIAL (+{:g}) '
                          '({} STRUCTURE violation(s))'.format(s1, len(structural)))
            for f, m in structural[:5]:
                report.append('       - {} : {}'.format(f, m[:60]))
        else:
            report.append('[S1] grounding structure (0 violations) OK  (+12)')

    # ---------------- S2: problem detection on YOUR data (8) ----------------
    checks = [
        (flagged(ta, 'ik_target_mid', 'ARM_OUT_OF_RANGE'), 2,
         'target_mid flagged ARM_OUT_OF_RANGE'),
        (flagged(ta, 'ik_target_far', 'ARM_OUT_OF_RANGE'), 2,
         'target_far flagged ARM_OUT_OF_RANGE'),
        (not any('ik_target_near' in f and not m.startswith('STRUCTURE')
                 for f, m in ta), 2,
         'target_near clean (no problem flags)'),
        (not any(m.startswith('JOINT_LIMIT_VIOLATION') for _, m in ta), 2,
         'no JOINT_LIMIT_VIOLATION'),
    ]
    for ok, pts, label in checks:
        ok = ok and ta_valid   # no valid report -> no vacuous passes
        total += pts if ok else 0
        report.append('[S2] {} {} (+{})'.format(
            (label + ' ').ljust(40, '.'), 'OK ' if ok else 'FAIL',
            pts if ok else 0))

    # ------- S3: your shapes vs TA dataset, per-case answer-key match (10) -------
    probe = parse_report('probe-validation.ttl')
    key_path = os.path.join(HERE, 'ta-answer-key.json')
    if probe is None or not os.path.exists(key_path):
        report.append('[S3] probe-validation.ttl missing/invalid or answer key missing  FAIL (+0)')
    else:
        with open(key_path) as f:
            answer_key = json.load(f)
        # Student report: collect each case's flag set from message prefixes.
        found = {case: set() for case in answer_key}
        for focus, msg in probe:
            prefix = msg.split(':')[0].strip()
            for case in answer_key:
                if case in focus:
                    found[case].add(prefix)
        matched = [c for c in sorted(answer_key)
                   if found[c] == set(answer_key[c])]
        mismatched = [c for c in sorted(answer_key) if c not in matched]
        faulty = [c for c in answer_key if answer_key[c]]
        clean = [c for c in answer_key if not answer_key[c]]
        hit_f = sum(1 for c in faulty if c in matched)
        hit_c = sum(1 for c in clean if c in matched)
        # every probe record weighs the same: a wrong boundary case or a false
        # positive costs as much as a missed fault
        s3 = round(10.0 * len(matched) / len(answer_key), 1)
        total += s3
        status = 'OK ' if not mismatched else 'PARTIAL'
        report.append('[S3] TA dataset vs answer key: faulty {}/{}, clean {}/{} '
                      '{} (+{:g})'.format(hit_f, len(faulty), hit_c, len(clean),
                                          status, s3))
        for c in mismatched[:6]:
            report.append('       - {}: expected {} , got {}'.format(
                c, sorted(answer_key[c]) or ['(clean)'],
                sorted(found[c]) or ['(clean)']))

    # ---------------- summary ----------------
    print('=' * 70)
    print('  Task 1 : Semantic Grounding (REUSE) + SHACL Validation')
    print('=' * 70)
    for line in report:
        print('  ' + line)
    print('-' * 70)
    print('  Your Task 1 Score : {:.1f} / 30.0'.format(total))
    print('=' * 70)
    sys.exit(0)  # a partial score is a result, not a crash


if __name__ == '__main__':
    main()
