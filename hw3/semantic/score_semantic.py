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
    results = []
    for chunk in RESULT_SPLIT.split(txt)[1:]:
        focus = re.search(r'sh:focusNode\s+([^\s;]+)', chunk)
        msg = re.search(r'sh:resultMessage\s+"([^"]*)"', chunk)
        results.append((focus.group(1) if focus else '',
                        msg.group(1) if msg else ''))
    return results


def flagged(results, focus_key, msg_prefix=None):
    for focus, msg in results:
        if focus_key in focus and (msg_prefix is None
                                   or msg.startswith(msg_prefix)):
            return True
    return False


def main():
    total = 0.0
    report = []

    # ---------------- S1: grounding structure (10) ----------------
    ta = parse_report('ta-validation.ttl')
    if ta is None:
        report.append('[S1] ta-validation.ttl missing ........ FAIL (+0)')
        ta = []
    else:
        structural = [(f, m) for f, m in ta if m.startswith('STRUCTURE')]
        s1 = max(0.0, 12.0 - 2.0 * len(structural))
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
        total += pts if ok else 0
        report.append('[S2] {} {} (+{})'.format(
            (label + ' ').ljust(40, '.'), 'OK ' if ok else 'FAIL',
            pts if ok else 0))

    # ------- S3: your shapes vs TA dataset, per-case answer-key match (10) -------
    probe = parse_report('probe-validation.ttl')
    key_path = os.path.join(HERE, 'ta-answer-key.json')
    if probe is None or not os.path.exists(key_path):
        report.append('[S3] probe-validation.ttl / answer key missing  FAIL (+0)')
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
        s3 = round(8.0 * hit_f / len(faulty) + 2.0 * hit_c / len(clean), 1)
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
    sys.exit(0 if total >= 29.9 else 1)


if __name__ == '__main__':
    main()
