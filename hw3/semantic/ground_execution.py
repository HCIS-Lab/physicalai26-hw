# -*- coding: utf-8 -*-
"""Grounding: turn the FK / IK execution process into semantic data (TA-provided).

This file is COMPLETE — there is nothing to implement here. It runs YOUR
your_fk / your_ik (Task 1 and Task 2) on a few cases and writes what happened
as a knowledge graph in semantic/output/Student.ttl:

  * the robot specification (cora:Robot, 6 soma:RevoluteJoint with D-H values)
  * 3 FK computations; every FK computation lists its 6 multiplication steps
    (hw3:FKStep: local transform T_k and cumulative transform A_k = A_(k-1) @ T_k)
  * 3 IK computations (status, residual, target distance, iteration budget, joint states)

Task 4 then compares this Student.ttl with the TA's execution (TA.ttl).

Vocabulary: IEEE 1872 CORA (robots), SOMA (joints / joint states / poses),
IEEE 1872 POS (positions), QUDT (units); local stubs are in
ontology/hw3-ontology.ttl.

Run (inside the pixi environment):
    pixi run task4                                   # grounding + full Task 4 scoring
    python semantic/ground_execution.py --group <id> # grounding only
"""

import inspect
import json
import os

import numpy as np

import common
from common import HOME_JOINTS, HW_ROOT, JOINT_LIMITS, SHARED_TARGETS, \
    SHOULDER_HEIGHT
from fk_trace import trace_fk

from fk import your_fk, get_ur5_DH_params  # noqa: E402
from ik import your_ik, pybullet_ik  # noqa: E402

OUTPUT_FILE = 'Student.ttl'
NUM_FK_CASES = 3  # number of FK test cases to evaluate and ground

GROUP_ID = 'student-group-00'


# ---------------------------------------------------------------------------
# One FK computation (+ its multiplication steps) -> triples
# ---------------------------------------------------------------------------
def fk_computation_to_triples(case_index, joint_config, eef_pose_7d,
                              pose_error, jacobian_error, steps=None):
    """Ground one FK execution: the input joint configuration, the computed
    EE pose, the errors vs ground truth and (optionally) the multiplication
    steps of the chain.

        stu:fk_case_0 a hw3:FKComputation ;
            hw3:caseIndex 0 ;                      # only for integer case ids
            hw3:producedBy "<group id>" ;
            hw3:computedForRobot stu:my_ur5 ;
            hw3:hasInputJointConfiguration stu:fk_case_0_q ;
            hw3:hasEndEffectorPose stu:fk_case_0_ee ;
            hw3:hasPoseError "0.00062"^^xsd:double ;     # position error (m)
            hw3:hasJacobianError "0.00031"^^xsd:double ; # -1.0 = not computed
            hw3:hasStep stu:fk_case_0_step0 , ... , stu:fk_case_0_step6 .
        stu:fk_case_0_step3 a hw3:FKStep ;
            hw3:stepIndex 3 ;
            hw3:isStepOfComputation stu:fk_case_0 ;
            hw3:hasLocalTransform stu:fk_case_0_step3_T ;       # T_3
            hw3:hasCumulativeTransform stu:fk_case_0_step3_A .  # A_3 = A_2 @ T_3
        (T / A are hw3:HomogeneousTransform nodes with entries hw3:m00 .. hw3:m23)

    Step 0 is the base: its cumulative transform is the base pose.
    """
    node = 'stu:fk_case_{}'.format(case_index)
    lines = ['{} a hw3:FKComputation ;'.format(node)]
    if isinstance(case_index, (int, np.integer)):
        lines.append('    hw3:caseIndex {} ;'.format(int(case_index)))
    lines += [
        '    hw3:producedBy "{}" ;'.format(GROUP_ID),
        '    hw3:computedForRobot stu:my_ur5 ;',
        '    hw3:hasInputJointConfiguration {}_q ;'.format(node),
        '    hw3:hasEndEffectorPose {}_ee ;'.format(node),
        '    hw3:hasPoseError "{}"^^xsd:double ;'.format(pose_error),
    ]
    if steps:
        lines.append('    hw3:hasJacobianError "{}"^^xsd:double ;'.format(
            jacobian_error))
        lines.append('    hw3:hasStep {} .'.format(' , '.join(
            '{}_step{}'.format(node, s['index']) for s in steps)))
    else:
        lines.append('    hw3:hasJacobianError "{}"^^xsd:double .'.format(
            jacobian_error))
    lines.append('')
    lines += common.joint_config_to_ttl('{}_q'.format(node), joint_config)
    lines += common.pose_to_ttl('{}_ee'.format(node), eef_pose_7d)
    for s in steps or []:
        step = '{}_step{}'.format(node, s['index'])
        lines += [
            '{} a hw3:FKStep ;'.format(step),
            '    hw3:stepIndex {} ;'.format(s['index']),
            '    hw3:isStepOfComputation {} ;'.format(node),
            '    hw3:hasLocalTransform {}_T ;'.format(step),
            '    hw3:hasCumulativeTransform {}_A .'.format(step),
            '',
        ]
        lines += common.matrix_to_ttl('{}_T'.format(step), s['local'])
        lines += common.matrix_to_ttl('{}_A'.format(step), s['cumulative'])
    return lines


# ---------------------------------------------------------------------------
# Robot specification -> triples
# ---------------------------------------------------------------------------
def robot_spec_to_triples(dh_params, joint_limits):
    """stu:my_ur5 a cora:Robot (6 DoF) and stu:my_ur5_joint1..6, each a
    soma:RevoluteJoint with its D-H parameters and joint limits."""
    joints = ' , '.join('stu:my_ur5_joint{}'.format(i + 1) for i in range(6))
    lines = [
        'stu:my_ur5 a cora:Robot ;',
        '    hw3:producedBy "{}" ;'.format(GROUP_ID),
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


# ---------------------------------------------------------------------------
# One IK computation -> triples
# ---------------------------------------------------------------------------
def ik_computation_to_triples(target_uri, joint_config, ik_status, residual,
                              target_distance, max_iterations=None):
    """stu:ik_<target> a hw3:IKComputation with status, residual (m), target
    distance (m), the resulting joint configuration and (optionally) the
    iteration budget hw3:hasMaxIterations the solver ran with."""
    node = 'stu:ik_{}'.format(target_uri)
    lines = [
        '{} a hw3:IKComputation ;'.format(node),
        '    hw3:producedBy "{}" ;'.format(GROUP_ID),
        '    hw3:computedForRobot stu:my_ur5 ;',
        '    hw3:solvesForTarget hw3:{} ;'.format(target_uri),
        '    hw3:hasIKStatus "{}" ;'.format(ik_status),
        '    hw3:hasResidual "{}"^^xsd:double ;'.format(residual),
        '    hw3:hasTargetDistance "{}"^^xsd:double ;'.format(target_distance),
    ]
    if max_iterations is not None:
        lines.append('    hw3:hasMaxIterations {} ;'.format(int(max_iterations)))
    lines += [
        '    hw3:hasJointConfiguration {}_q .'.format(node),
        '',
    ]
    lines += common.joint_config_to_ttl('{}_q'.format(node), joint_config)
    return lines


# ---------------------------------------------------------------------------
# Shared-target instances (world frame, structured)
# ---------------------------------------------------------------------------
def targets_to_triples():
    lines = ['# Shared IK targets (fixed experiment settings)', '']
    for target in SHARED_TARGETS:
        lines += common.pose_to_ttl('hw3:{}'.format(target['uri']),
                                    list(target['position']))
    return lines


# ---------------------------------------------------------------------------
# Execution flow (FK part + IK part -> one Student.ttl)
# ---------------------------------------------------------------------------
def main():
    args = common.make_arg_parser(__doc__).parse_args()
    global GROUP_ID
    GROUP_ID = args.group

    dh_params = get_ur5_DH_params()
    lines = targets_to_triples()
    lines += robot_spec_to_triples(dh_params, JOINT_LIMITS)

    # ---- FK execution process (3 cases vs course ground truth) ----
    with open(os.path.join(HW_ROOT, 'test_case',
                           'fk_test_case_easy.json')) as f:
        gt = json.load(f)
    robot, base_pos = common.connect_sim()
    for i, q in enumerate(gt['joint_poses'][:NUM_FK_CASES]):
        steps = trace_fk(dh_params, q, base_pos)   # correct multiplication
        if not args.reference:
            pose, _jac = your_fk(dh_params, q, base_pos)
            jacobian_error = round(float(np.linalg.norm(
                np.asarray(_jac) - np.asarray(gt['jacobian'][i]))), 6)
            chain_gap = float(np.linalg.norm(
                np.asarray(pose)[:3] - steps[-1]['cumulative'][:3, 3]))
            if chain_gap > 0.005:
                print('[FK] WARNING case {}: your_fk differs from the reference '
                      'chain by {:.4f} m (Student.ttl records the reference '
                      'multiplication steps)'.format(i, chain_gap))
        else:
            # debug: read the pose from the simulator; Jacobian not computed
            pose = list(common.sim_eef_pose(robot, q))
            jacobian_error = -1.0
        # FK error is a POSITION error in metres.
        pose_error = round(float(np.linalg.norm(
            np.asarray(pose)[:3] - np.asarray(gt['poses'][i])[:3])), 6)
        print('[FK] case {} -> pose_err={:.6f} m jac_err={:.6f}'.format(
            i, pose_error, jacobian_error))
        lines += fk_computation_to_triples(i, q, list(pose), pose_error,
                                           jacobian_error, steps)

    # ---- IK execution process (3 shared targets) ----
    home_quat = np.asarray(robot.get_eef_pose())[3:]
    # iteration budget of the solver that produced the result (default argument
    # of your_ik: the grader only uses default arguments)
    max_iters = inspect.signature(your_ik).parameters['max_iters'].default
    shoulder = np.asarray(base_pos) + [0.0, 0.0, SHOULDER_HEIGHT]
    for target in SHARED_TARGETS:
        target_pose_7d = list(target['position']) + list(home_quat)
        common.reset_arm(robot, HOME_JOINTS)
        if not args.reference:
            joints = list(your_ik(robot.robot_id, target_pose_7d,
                                  base_pos=base_pos))
        else:
            joints = list(np.asarray(
                pybullet_ik(robot.robot_id, target_pose_7d))[:6])
            joints = [min(max(j, lo), hi)
                      for j, (lo, hi) in zip(joints, JOINT_LIMITS)]
        # judge the solution with the SIMULATOR pose, not with your own FK
        eef_pos = common.sim_eef_pose(robot, joints)[:3]
        status, residual = common.classify_ik_result(
            eef_pos, target['position'], base_pos)
        distance = round(float(np.linalg.norm(
            np.asarray(target['position']) - shoulder)), 5)
        print('[IK] {} -> status={}, residual={:.4f} m, distance={:.3f} m'
              .format(target['uri'], status, residual, distance))
        lines += ik_computation_to_triples(
            target['uri'], joints, status, round(residual, 5), distance,
            max_iters)

    common.write_graph(OUTPUT_FILE, lines, GROUP_ID, 'ground_execution.py')

    import pybullet as p
    p.disconnect()


if __name__ == '__main__':
    main()
