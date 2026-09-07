# -*- coding: utf-8 -*-
"""Shared constants and helpers for the Task 1 grounding scripts (TA-provided).

The three per-task grounding scripts (ground_task1_fk.py, ground_task2_ik.py,
ground_task3_insertion.py) each produce their own knowledge graph, but they
all speak the same vocabulary <http://taica.course/hw3/ontology#> — that is
what lets one triple store and one set of SPARQL queries evaluate all three
tasks together.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HW_ROOT = os.path.abspath(os.path.join(HERE, '..'))
if HW_ROOT not in sys.path:
    sys.path.insert(0, HW_ROOT)

HW3_NS = 'http://taica.course/hw3/ontology#'


def stu_ns(student_id):
    """Per-student data namespace: http://taica.course/hw3/data/<student-id>#

    This is an INDIVIDUAL assignment: every student publishes their
    instances under their own student-ID namespace while sharing the hw3:
    vocabulary and the shared target URIs. Any number of submissions can
    therefore be loaded into one triple store without URI collisions, and
    one SPARQL query can compare every student's results (see q3, which
    generalizes from 2 robots to N robots automatically).
    """
    safe = ''.join(c if (c.isalnum() or c in '-_') else '-' for c in student_id)
    return 'http://taica.course/hw3/data/{}#'.format(safe)

OUTPUT_DIR = os.path.join(HERE, 'output')

SIM_TIMESTEP = 1.0 / 240.0
UR5_MAX_REACH = 0.90          # UR5 workspace radius (m); beyond => OUT_OF_REACH
SHOULDER_HEIGHT = 0.0892      # DH d1: shoulder height (m)
IK_SOLVED_THRESH = 0.02       # residual threshold for SOLVED (m)
FK_POSE_ERROR_THRESH = 0.005  # Task 1 pose error threshold (course value)
FK_JACOBIAN_ERROR_THRESH = 0.05  # Task 1 Jacobian error threshold (course value)

# Joint limits consistent with ik.py; grounded into triples in Task 1
JOINT_LIMITS = [
    [-3 * np.pi / 2, -np.pi / 2],   # joint1
    [-2.3562, -1.0],                # joint2
    [-17.0, 17.0],                  # joint3
    [-17.0, 17.0],                  # joint4
    [-17.0, 17.0],                  # joint5
    [-17.0, 17.0],                  # joint6
]

# Shared targets (exactly the same coordinates as hw3:target_* in the
# ontology; the URIs must also be the same set, otherwise SPARQL cannot
# join across graphs)
SHARED_TARGETS = [
    {'uri': 'target_near', 'position': [0.35, 0.28, 0.95]},
    {'uri': 'target_mid',  'position': [0.90, 0.13, 0.80]},
    {'uri': 'target_far',  'position': [1.80, 0.50, 0.95]},
]

HOME_JOINTS = (np.array([-1.0, -0.5, 0.5, -0.5, -0.5, 0.0]) * np.pi).tolist()


def make_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--student-id', default='b00000000',
                        help='your student ID — becomes your data namespace http://taica.course/hw3/data/<student-id># and the hw3:producedBy value')
    parser.add_argument('--own', action='store_true',
                        help='use YOUR your_fk/your_ik to generate the '
                             'execution data (rerun after Tasks 2-3); the '
                             'default uses the TA reference solvers so '
                             'Task 1 works before any implementation')
    return parser


def connect_sim():
    """Start a DIRECT pybullet session with the course UR5; returns (robot, base_pos)."""
    import pybullet as p
    from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env

    physics_client_id = p.connect(p.DIRECT)
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)

    robot = ur5Env(physics_client_id, use_IK=1)
    for _ in range(240):
        p.stepSimulation()
    return robot, list(robot._base_position)


def reset_arm(robot, joint_values):
    import pybullet as p
    joint_ids = list(robot._joint_name_to_ids.values())
    for jid, val in zip(joint_ids, joint_values):
        p.resetJointState(robot.robot_id, jid, val)


def sim_eef_pose(robot, joint_values):
    """Read the end-effector pose at given joint angles without side effects
    (reset -> read -> restore)."""
    import pybullet as p
    joint_ids = list(robot._joint_name_to_ids.values())
    saved = [p.getJointState(robot.robot_id, j)[0] for j in joint_ids]
    reset_arm(robot, joint_values)
    pose = np.asarray(robot.get_eef_pose())
    reset_arm(robot, saved)
    return pose


def classify_ik_result(eef_pos_after_ik, target_pos, base_pos):
    """Determine the IK status string from residual and workspace geometry."""
    target_pos = np.asarray(target_pos, dtype=float)
    shoulder = np.asarray(base_pos, dtype=float) + [0.0, 0.0, SHOULDER_HEIGHT]
    dist = float(np.linalg.norm(target_pos - shoulder))
    residual = float(np.linalg.norm(
        np.asarray(eef_pos_after_ik, dtype=float) - target_pos))

    if residual < IK_SOLVED_THRESH:
        return 'SOLVED', residual
    if dist > UR5_MAX_REACH:
        return 'OUT_OF_REACH', residual
    return 'NO_CONVERGENCE', residual


def write_graph(filename, lines, student_id, script_name):
    """Write a Turtle knowledge graph with the shared prefixes.

    The stu: prefix is bound to THIS student's namespace, so the same
    triple-writing code (stu:my_ur5, stu:ik_target_near, ...) yields
    student-unique URIs — submissions from different students can be merged
    into one store and cross-queried.
    """
    header = [
        '@prefix hw3:  <{}> .'.format(HW3_NS),
        '@prefix stu:  <{}> .'.format(stu_ns(student_id)),
        '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
        '@prefix cora: <http://purl.org/ieee1872-owl/cora-bare#> .',
        '@prefix soma: <http://www.ease-crc.org/ont/SOMA.owl#> .',
        '@prefix pos:  <http://purl.org/ieee1872-owl/pos#> .',
        '@prefix qudt: <http://qudt.org/schema/qudt/> .',
        '@prefix unit: <http://qudt.org/vocab/unit/> .',
        '@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .',
        '',
        '# Auto-generated by semantic/{}'.format(script_name),
        '# producedBy: {}'.format(student_id),
        '',
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(header + lines) + '\n')
    print('[GROUND] knowledge graph written to {}'.format(path))
    return path
# ---------------------------------------------------------------------------
# Structured-serialization helpers (unified representation, no JSON strings)
#
# Poses / joint configurations / trajectories are STRUCTURED RDF nodes that
# reuse existing robot ontologies (all stubbed in ontology/hw3-ontology.ttl):
#   - soma:6DPose + IEEE 1872 pos:Position  -> pose = position + quaternion
#   - soma:JointState (hw3:isStateOfJoint)  -> one state per joint individual
#   - QUDT unit IRIs (unit:M / unit:RAD)     -> units never live in strings
# ---------------------------------------------------------------------------

def _double(value):
    """Format a float as a typed xsd:double Turtle literal."""
    return '"{}"^^xsd:double'.format(round(float(value), 6))


def pose_to_ttl(node, pose, frame='world'):
    """Serialize a 7d pose [x, y, z, qx, qy, qz, qw] (3d accepted; identity
    quaternion assumed) as a structured soma:6DPose node."""
    x, y, z = (float(v) for v in pose[:3])
    quat = ([float(v) for v in pose[3:7]] if len(pose) >= 7
            else [0.0, 0.0, 0.0, 1.0])
    return [
        '{} a soma:6DPose ;'.format(node),
        '    hw3:hasReferenceFrame "{}" ;'.format(frame),
        '    hw3:hasPositionComponent {}_pos ;'.format(node),
        '    hw3:hasOrientationComponent {}_ori .'.format(node),
        '{}_pos a pos:Position ;'.format(node),
        '    qudt:hasUnit unit:M ;',
        '    hw3:hasPositionX {} ;'.format(_double(x)),
        '    hw3:hasPositionY {} ;'.format(_double(y)),
        '    hw3:hasPositionZ {} .'.format(_double(z)),
        '{}_ori a hw3:QuaternionOrientation ;'.format(node),
        '    hw3:hasQuatX {} ;'.format(_double(quat[0])),
        '    hw3:hasQuatY {} ;'.format(_double(quat[1])),
        '    hw3:hasQuatZ {} ;'.format(_double(quat[2])),
        '    hw3:hasQuatW {} .'.format(_double(quat[3])),
        '',
    ]


def joint_config_to_ttl(node, q, joint_node_fmt='stu:my_ur5_joint{}'):
    """Serialize a 6-DoF joint vector as hw3:JointConfiguration — one
    soma:JointState per joint, each pointing back at the joint individual."""
    state_nodes = ['{}_js{}'.format(node, i + 1) for i in range(len(q))]
    lines = [
        '{} a hw3:JointConfiguration ;'.format(node),
        '    hw3:hasJointState {} .'.format(' , '.join(state_nodes)),
    ]
    for i, (angle, state_node) in enumerate(zip(q, state_nodes)):
        lines += [
            '{} a soma:JointState ;'.format(state_node),
            '    hw3:isStateOfJoint {} ;'.format(joint_node_fmt.format(i + 1)),
            '    qudt:hasUnit unit:RAD ;',
            '    soma:hasJointPosition {} .'.format(_double(angle)),
        ]
    lines.append('')
    return lines
