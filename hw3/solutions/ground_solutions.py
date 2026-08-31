# -*- coding: utf-8 -*-
"""TA reference solution (DO NOT DISTRIBUTE): implements the STUDENT TODO
functions of semantic/ground_execution.py and runs it.

For the shapes.ttl STUDENT TODO, copy the solution file over first:
    cp solutions/shapes_solution.ttl     hw3_template/semantic/shapes.ttl

Usage (invoked by run_task1.sh via the GROUND_SCRIPT override):
    cd hw3_template/semantic
    GROUND_SCRIPT=/path/to/this/file PYTHON=... bash run_task1.sh
"""

import os
import sys

SEMANTIC_DIR = os.getcwd()  # run_task1.sh cd's into semantic/
sys.path.insert(0, SEMANTIC_DIR)

import common  # noqa: E402
import ground_execution as ge  # noqa: E402


def _prov():
    """Provenance id: the template uses GROUP_ID, the demo STUDENT_ID."""
    return getattr(ge, 'GROUP_ID', None) or getattr(ge, 'STUDENT_ID', 'ta')


def robot_spec_to_triples(dh_params, joint_limits):
    joints = ' , '.join('stu:my_ur5_joint{}'.format(i + 1) for i in range(6))
    lines = [
        'stu:my_ur5 a cora:Robot ;',
        '    hw3:producedBy "{}" ;'.format(_prov()),
        '    hw3:hasDoF 6 ;',
        '    hw3:hasJoint {} .'.format(joints),
        '',
    ]
    for i, (dh, lim) in enumerate(zip(dh_params, joint_limits)):
        lines += [
            'stu:my_ur5_joint{} a soma:RevoluteJoint ;'.format(i + 1),
            '    hw3:jointIndex {} ;'.format(i + 1),
            '    hw3:dh_a "{}"^^xsd:double ;'.format(round(float(dh['a']), 6)),
            '    hw3:dh_d "{}"^^xsd:double ;'.format(round(float(dh['d']), 6)),
            '    hw3:dh_alpha "{}"^^xsd:double ;'.format(
                round(float(dh['alpha']), 6)),
            '    hw3:hasJointLowerLimit "{}"^^xsd:double ;'.format(
                round(float(lim[0]), 6)),
            '    hw3:hasJointUpperLimit "{}"^^xsd:double .'.format(
                round(float(lim[1]), 6)),
            '',
        ]
    return lines


def ik_computation_to_triples(target_uri, joint_config, ik_status, residual,
                              target_distance):
    node = 'stu:ik_{}'.format(target_uri)
    lines = [
        '{} a hw3:IKComputation ;'.format(node),
        '    hw3:producedBy "{}" ;'.format(_prov()),
        '    hw3:computedForRobot stu:my_ur5 ;',
        '    hw3:solvesForTarget hw3:{} ;'.format(target_uri),
        '    hw3:hasIKStatus "{}" ;'.format(ik_status),
        '    hw3:hasResidual "{}"^^xsd:double ;'.format(residual),
        '    hw3:hasTargetDistance "{}"^^xsd:double ;'.format(target_distance),
        '    hw3:hasJointConfiguration {}_q .'.format(node),
        '',
    ]
    lines += common.joint_config_to_ttl('{}_q'.format(node), joint_config)
    return lines


ge.robot_spec_to_triples = robot_spec_to_triples
ge.ik_computation_to_triples = ik_computation_to_triples

if __name__ == '__main__':
    ge.main()
