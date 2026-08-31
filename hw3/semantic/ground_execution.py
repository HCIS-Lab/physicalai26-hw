# -*- coding: utf-8 -*-
"""Task 1 / REUSE — Ground an FK/IK execution process into semantic data.

This task comes FIRST: by default the execution data is produced by the TA
reference solvers (ground-truth FK poses + the simulator IK), so no student
FK/IK implementation is needed. After finishing Tasks 2-3 re-run with --own
to ground YOUR solvers' execution instead.

把 FK/IK 的「執行過程資訊」轉進語意空間:
REUSE 既有標準本體 (IEEE 1872 CORA robots, SOMA joints/joint-states/poses,
IEEE 1872 POS positions, QUDT units — 存根都在 ontology/hw3-ontology.ttl),
把數值轉成結構化 RDF -> semantic/output/data.ttl。

之後 SHACL 驗證 (shapes.ttl) 會直接讀這份 data.ttl,從數值中發現
執行過程的問題狀態 (ARM_OUT_OF_RANGE / NO_CONVERGENCE)。

STUDENT TODO in this file:
    robot_spec_to_triples()      — 機器人規格 grounding
    ik_computation_to_triples()  — IK 執行結果 grounding
(fk_computation_to_triples() is fully implemented as the worked example.)

Run (inside the taica-hw3 conda environment):
    python semantic/ground_execution.py --student-id <your student ID>
    python semantic/ground_execution.py --student-id ... --own   # after Tasks 2-3
"""

import json
import os

import numpy as np

import common
from common import HOME_JOINTS, HW_ROOT, JOINT_LIMITS, SHARED_TARGETS, \
    SHOULDER_HEIGHT

from fk import your_fk, get_ur5_DH_params  # noqa: E402
from ik import your_ik, pybullet_ik  # noqa: E402

OUTPUT_FILE = 'data.ttl'
NUM_FK_CASES = 3  # number of FK test cases to evaluate and ground

STUDENT_ID = 'b00000000'


# ---------------------------------------------------------------------------
# Worked example (already implemented): one FK computation -> triples
# ---------------------------------------------------------------------------
def fk_computation_to_triples(case_index, joint_config, eef_pose_7d,
                              pose_error, jacobian_error):
    """(Example, complete) Ground one FK execution: the input joint
    configuration, the computed EE pose, and the errors vs ground truth.

    Produces (STRUCTURED representation — no JSON-string literals):
        stu:fk_case_0 a hw3:FKComputation ;
            hw3:producedBy "<student id>" ;
            hw3:computedForRobot stu:my_ur5 ;
            hw3:hasInputJointConfiguration stu:fk_case_0_q ;   # object node
            hw3:hasEndEffectorPose stu:fk_case_0_ee ;          # object node
            hw3:hasPoseError "0.00062"^^xsd:double ;
            hw3:hasJacobianError "0.00031"^^xsd:double .
        (+ the structured JointConfiguration / 6DPose nodes, built with
         common.joint_config_to_ttl / common.pose_to_ttl)

    Conventions your own functions must follow as well:
      1. Instances live in the stu: namespace; classes/properties in hw3:
      2. Numeric values are typed literals on STRUCTURED nodes — never
         JSON-string literals; units are QUDT IRIs (unit:M / unit:RAD)
      3. Every instance carries hw3:producedBy for provenance
    """
    node = 'stu:fk_case_{}'.format(case_index)
    lines = [
        '{} a hw3:FKComputation ;'.format(node),
        '    hw3:producedBy "{}" ;'.format(STUDENT_ID),
        '    hw3:computedForRobot stu:my_ur5 ;',
        '    hw3:hasInputJointConfiguration {}_q ;'.format(node),
        '    hw3:hasEndEffectorPose {}_ee ;'.format(node),
        '    hw3:hasPoseError "{}"^^xsd:double ;'.format(pose_error),
        '    hw3:hasJacobianError "{}"^^xsd:double .'.format(jacobian_error),
        '',
    ]
    lines += common.joint_config_to_ttl('{}_q'.format(node), joint_config)
    lines += common.pose_to_ttl('{}_ee'.format(node), eef_pose_7d)
    return lines


# ---------------------------------------------------------------------------
# STUDENT TODO 1: robot specification -> triples
# ---------------------------------------------------------------------------
def robot_spec_to_triples(dh_params, joint_limits):
    """Ground the UR5 specification; return Turtle lines (list of str).

    Must include:
      1. stu:my_ur5 a cora:Robot (IEEE 1872 CORA — reused, NOT an hw3:
         invention), with hw3:producedBy "<STUDENT_ID>", hw3:hasDoF 6,
         and hw3:hasJoint linking the 6 joint instances
      2. stu:my_ur5_joint1 ~ joint6, each a soma:RevoluteJoint (reused
         from SOMA) with hw3:jointIndex (1~6), hw3:dh_a / dh_d / dh_alpha
         (xsd:double), hw3:hasJointLowerLimit / hasJointUpperLimit

    Hint: float literals MUST be "..."^^xsd:double (a bare 0.0892 in
    Turtle is xsd:decimal and fails the TA structural SHACL shapes).
    """
    lines = []

    # ------------------------- your code -------------------------

    joints = ' , '.join('stu:my_ur5_joint{}'.format(i + 1) for i in range(6))
    lines += [
        'stu:my_ur5 a cora:Robot ;',
        '    hw3:producedBy "{}" ;'.format(STUDENT_ID),
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

    # --------------------------------------------------------------

    return lines


# ---------------------------------------------------------------------------
# STUDENT TODO 2: one IK computation -> triples
# ---------------------------------------------------------------------------
def ik_computation_to_triples(target_uri, joint_config, ik_status, residual,
                              target_distance):
    """Ground one IK execution; return Turtle lines (list of str).

    Must include one instance (suggested URI: stu:ik_<target_uri>) of type
    hw3:IKComputation, carrying:
      - hw3:producedBy          "<STUDENT_ID>"
      - hw3:computedForRobot    stu:my_ur5
      - hw3:solvesForTarget     hw3:<target_uri>   <-- hw3: namespace!
      - hw3:hasIKStatus         "SOLVED" / "OUT_OF_REACH" / ... (string)
      - hw3:hasResidual         residual ("..."^^xsd:double)
      - hw3:hasTargetDistance   target_distance ("..."^^xsd:double)
                                — SHACL 的 reach 檢查讀這個欄位
      - hw3:hasJointConfiguration  a STRUCTURED hw3:JointConfiguration node
                                (6x soma:JointState via
                                common.joint_config_to_ttl)
    """
    lines = []

    # ------------------------- your code -------------------------

    node = 'stu:ik_{}'.format(target_uri)
    lines += [
        '{} a hw3:IKComputation ;'.format(node),
        '    hw3:producedBy "{}" ;'.format(STUDENT_ID),
        '    hw3:computedForRobot stu:my_ur5 ;',
        '    hw3:solvesForTarget hw3:{} ;'.format(target_uri),
        '    hw3:hasIKStatus "{}" ;'.format(ik_status),
        '    hw3:hasResidual "{}"^^xsd:double ;'.format(residual),
        '    hw3:hasTargetDistance "{}"^^xsd:double ;'.format(target_distance),
        '    hw3:hasJointConfiguration {}_q .'.format(node),
        '',
    ]
    lines += common.joint_config_to_ttl('{}_q'.format(node), joint_config)

    # --------------------------------------------------------------

    return lines


# ---------------------------------------------------------------------------
# TA-provided: shared-target instances (world frame, structured)
# ---------------------------------------------------------------------------
def targets_to_triples():
    lines = ['# Shared IK targets (fixed experiment settings)', '']
    for target in SHARED_TARGETS:
        lines += common.pose_to_ttl('hw3:{}'.format(target['uri']),
                                    list(target['position']))
    return lines


# ---------------------------------------------------------------------------
# TA-provided: execution flow (FK part + IK part -> one data.ttl)
# ---------------------------------------------------------------------------
def main():
    args = common.make_arg_parser(__doc__).parse_args()
    global STUDENT_ID
    STUDENT_ID = args.student_id

    dh_params = get_ur5_DH_params()
    lines = targets_to_triples()
    lines += robot_spec_to_triples(dh_params, JOINT_LIMITS)

    # ---- FK execution process (3 cases vs course ground truth) ----
    with open(os.path.join(HW_ROOT, 'test_case',
                           'fk_test_case_easy.json')) as f:
        gt = json.load(f)
    base_pos = [-0.2, 0.13, 0.6]  # ur5Env default base position
    for i, q in enumerate(gt['joint_poses'][:NUM_FK_CASES]):
        if args.own:
            pose, _jac = your_fk(dh_params, q, base_pos)
            pose_error = round(float(np.linalg.norm(
                np.asarray(pose) - np.asarray(gt['poses'][i]))), 6)
            jacobian_error = round(float(np.linalg.norm(
                np.asarray(_jac) - np.asarray(gt['jacobian'][i]))), 6)
        else:
            # Task 1 default: TA reference execution — the course ground
            # truth IS the reference FK, so the errors are zero.
            pose = list(gt['poses'][i])
            pose_error = 0.0
            jacobian_error = 0.0
        print('[FK] case {} -> pose_err={:.6f} jac_err={:.6f}'.format(
            i, pose_error, jacobian_error))
        lines += fk_computation_to_triples(i, q, list(pose),
                                           pose_error, jacobian_error)

    # ---- IK execution process (3 shared targets) ----
    robot, base_pos = common.connect_sim()
    home_quat = np.asarray(robot.get_eef_pose())[3:]
    shoulder = np.asarray(base_pos) + [0.0, 0.0, SHOULDER_HEIGHT]
    for target in SHARED_TARGETS:
        target_pose_7d = list(target['position']) + list(home_quat)
        common.reset_arm(robot, HOME_JOINTS)
        if args.own:
            joints = list(your_ik(robot.robot_id, target_pose_7d,
                                  base_pos=base_pos))
            eef_pos, _ = your_fk(dh_params, joints, base_pos)
            eef_pos = eef_pos[:3]
        else:
            # Task 1 default: TA reference IK (simulator solver),
            # clipped to the course joint limits exactly like your_ik.
            joints = list(np.asarray(
                pybullet_ik(robot.robot_id, target_pose_7d))[:6])
            joints = [min(max(j, lo), hi)
                      for j, (lo, hi) in zip(joints, JOINT_LIMITS)]
            eef_pos = common.sim_eef_pose(robot, joints)[:3]
        status, residual = common.classify_ik_result(
            eef_pos, target['position'], base_pos)
        distance = round(float(np.linalg.norm(
            np.asarray(target['position']) - shoulder)), 5)
        print('[IK] {} -> status={}, residual={:.4f} m, distance={:.3f} m'
              .format(target['uri'], status, residual, distance))
        lines += ik_computation_to_triples(
            target['uri'], joints, status, round(residual, 5), distance)

    common.write_graph(OUTPUT_FILE, lines, STUDENT_ID, 'ground_execution.py')

    import pybullet as p
    p.disconnect()


if __name__ == '__main__':
    main()
