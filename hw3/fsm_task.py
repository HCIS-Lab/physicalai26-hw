# -*- coding: utf-8 -*-
"""Task 4 — Finite State Machine manipulation pipeline (TA-provided).

Validates YOUR IK and FK inside a complete pick-and-place pipeline —
no learned model, no dataset, no checkpoint: a deterministic FSM drives
the UR5 through 10 seeded episodes of "pick the block, place it on the
goal marker".

    MOVE_TO_PREPICK -> DESCEND_TO_PICK -> GRASP -> LIFT
        -> MOVE_TO_PREPLACE -> DESCEND_TO_PLACE -> RELEASE -> RETREAT
        -> VERIFY -> SUCCESS / FAILURE

Every motion state does two validations:
  * IK validation : the commanded pose is solved by `your_ik`, executed
    with position control, and the achieved end-effector position must
    land within IK_TOL of the command.
  * FK validation : `your_fk` is evaluated on the MEASURED joint angles
    and must agree with the simulator's end-effector position within
    FK_TOL — your kinematic model is checked against physics at every
    state transition.

Run:
    python fsm_task.py          # scoring mode (headless)
    python fsm_task.py -g       # watch it in the PyBullet GUI
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from fk import your_fk, get_ur5_DH_params
from ik import your_ik

SIM_TIMESTEP = 1.0 / 240.0
TASK4_SCORE_MAX = 10.0
NUM_EPISODES = 10

IK_TOL = 0.03       # m: achieved EE position vs commanded target
FK_TOL = 0.01       # m: your_fk(measured q) vs simulator EE position
PLACE_TOL = 0.06    # m: final block XY vs goal XY

TABLE_TOP_Z = 0.649        # cube_small resting height on the table
SAFE_Z = 0.95              # travel height
DESCEND_START_Z = 0.78     # descend begins here (well above the block)
DESCEND_STEP = 0.01        # m per descend increment
PICK_FLOOR_Z = 0.655       # lowest EE command while descending

STATES = ['MOVE_TO_PREPICK', 'DESCEND_TO_PICK', 'GRASP', 'LIFT',
          'MOVE_TO_PREPLACE', 'DESCEND_TO_PLACE', 'RELEASE', 'RETREAT',
          'VERIFY']


def sample_episode(seed):
    """Seeded block / goal positions, both well inside the UR5 workspace."""
    rng = np.random.RandomState(seed)
    while True:
        # sampled inside the workspace the Task 1/2 test cases already cover
        block = np.array([rng.uniform(0.32, 0.47), rng.uniform(0.02, 0.24)])
        goal = np.array([rng.uniform(0.32, 0.47), rng.uniform(0.02, 0.24)])
        if np.linalg.norm(block - goal) > 0.12:
            return block, goal


class FSMPipeline(object):
    """A deterministic pick-and-place state machine driven by your IK/FK."""

    def __init__(self, robot, base_pos, home_quat, gui=False):
        self.robot = robot
        self.base_pos = list(base_pos)
        self.home_quat = list(home_quat)
        self.gui = gui
        self.dh_params = get_ur5_DH_params()
        self.joint_ids = list(robot._joint_name_to_ids.values())
        self.log = []
        # Task 1 bridge: execution records for the SHACL diagnosis
        self.diag_fk, self.diag_ik, self._diag_n = [], [], 0

    # ---------------- low-level helpers ----------------
    def _step(self, seconds):
        for _ in range(int(seconds / SIM_TIMESTEP)):
            p.stepSimulation()
            if self.gui:
                time.sleep(SIM_TIMESTEP)

    def _measured_q(self):
        states = p.getJointStates(self.robot.robot_id, self.joint_ids)
        return [s[0] for s in states]

    def move_to(self, state, position, seconds=2.0):
        """One FSM motion: your_ik -> position control -> IK & FK checks.

        Long jumps may need re-solving: iterative IK is warm-started from
        the CURRENT joint state, so after the first motion a re-solve
        refines the solution (exactly how the continuous Task 2 trajectory
        converges). Up to 3 solve-and-drive attempts per state.
        """
        target = list(position) + self.home_quat
        for attempt in range(3):
            joints = your_ik(self.robot.robot_id, target,
                             base_pos=self.base_pos)
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot.robot_id, jointIndices=self.joint_ids,
                controlMode=p.POSITION_CONTROL, targetPositions=joints,
                positionGains=[0.2] * len(joints),
                velocityGains=[1] * len(joints))
            self._step(seconds if attempt == 0 else 0.8)
            ee = np.asarray(self.robot.get_eef_pose())[:3]
            ik_err = float(np.linalg.norm(ee - np.asarray(position)))
            if ik_err <= IK_TOL:
                break
        q_meas = self._measured_q()
        fk_pose, _ = your_fk(self.dh_params, q_meas, self.base_pos)
        fk_err = float(np.linalg.norm(np.asarray(fk_pose[:3]) - ee))
        self.log.append((state, ik_err, fk_err))

        # Task 1 bridge: failing motions (and the very first motion, as a
        # clean sample) are grounded and SHACL-validated after the episodes.
        shoulder = np.asarray(self.base_pos) + [0.0, 0.0, 0.0892]
        distance = float(np.linalg.norm(np.asarray(position) - shoulder))
        self._diag_n += 1
        tag = 'diag_{:02d}_{}'.format(self._diag_n, state.lower())
        if fk_err > FK_TOL:
            self.diag_fk.append((tag + '_fk', q_meas, list(fk_pose),
                                 fk_err, 0.0))
        if ik_err > IK_TOL:
            self.diag_ik.append((tag, q_meas, 'MISS', ik_err, distance))
        elif self._diag_n == 1:
            self.diag_ik.append((tag, q_meas, 'SOLVED', ik_err, distance))

        if fk_err > FK_TOL:
            return 'FK_MISMATCH in {} (fk vs sim = {:.4f} m)'.format(state, fk_err)
        if ik_err > IK_TOL:
            return 'IK_MISS in {} (achieved vs commanded = {:.4f} m)'.format(state, ik_err)
        return None

    # ---------------- the state machine ----------------
    def run_episode(self, block_id, block_xy, goal_xy):
        ee = self.robot.ee
        ee.obj_ids = {'rigid': [block_id]}

        transitions = [
            ('MOVE_TO_PREPICK', lambda: self.move_to(
                'MOVE_TO_PREPICK', [block_xy[0], block_xy[1], SAFE_Z])),
            ('DESCEND_TO_PICK', lambda: self._descend_until_contact(block_xy)),
            ('GRASP', lambda: self._grasp()),
            ('LIFT', lambda: self.move_to(
                'LIFT', [block_xy[0], block_xy[1], SAFE_Z])),
            ('MOVE_TO_PREPLACE', lambda: self.move_to(
                'MOVE_TO_PREPLACE', [goal_xy[0], goal_xy[1], SAFE_Z])),
            ('DESCEND_TO_PLACE', lambda: self.move_to(
                'DESCEND_TO_PLACE', [goal_xy[0], goal_xy[1], 0.78], 1.5)),
            ('RELEASE', lambda: self._release()),
            ('RETREAT', lambda: self.move_to(
                'RETREAT', [goal_xy[0], goal_xy[1], SAFE_Z])),
            ('VERIFY', lambda: self._verify(block_id, goal_xy)),
        ]
        for state, action in transitions:
            error = action()
            if error is not None:
                return False, state, error
        return True, 'VERIFY', None

    def _descend_until_contact(self, block_xy):
        z = DESCEND_START_Z
        while z > PICK_FLOOR_Z:
            err = self.move_to('DESCEND_TO_PICK',
                               [block_xy[0], block_xy[1], z], 0.4)
            if err is not None:
                return err
            if self.robot.ee.detect_contact():
                return None
            z -= DESCEND_STEP
        return 'NO_CONTACT in DESCEND_TO_PICK (suction never touched the block)'

    def _grasp(self):
        self.robot.ee.activate()
        self._step(0.2)
        if self.robot.ee.check_grasp() is None:
            return 'GRASP_FAILED (suction constraint not created)'
        return None

    def _release(self):
        self.robot.ee.release()
        self._step(0.5)
        return None

    def _verify(self, block_id, goal_xy):
        pos, _ = p.getBasePositionAndOrientation(block_id)
        err = float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(goal_xy)))
        self.place_error = err
        if err > PLACE_TOL:
            return 'PLACE_MISS (block {:.3f} m from goal)'.format(err)
        return None


def main(args):
    physics_client_id = p.connect(p.GUI if args.gui else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=90,
                                 cameraPitch=-25,
                                 cameraTargetPosition=[0.5, 0.1, 0.7])
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'table/table.urdf'),
               [0.9, 0.0, 0.0])

    from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env
    robot = ur5Env(physics_client_id, use_IK=1)
    for _ in range(240):
        p.stepSimulation()
    home_q = [p.getJointState(robot.robot_id, j)[0]
              for j in robot._joint_name_to_ids.values()]
    home_quat = np.asarray(robot.get_eef_pose())[3:]

    print('============================ Task 4 : FSM Manipulation Pipeline '
          '============================\n')
    print('FSM: ' + ' -> '.join(STATES) + '\n')

    fsm = FSMPipeline(robot, robot._base_position, home_quat, gui=args.gui)
    successes = 0
    for ep in range(NUM_EPISODES):
        block_xy, goal_xy = sample_episode(ep)

        # reset arm to home (reset + retarget motors + settle), then spawn
        for j, q in zip(robot._joint_name_to_ids.values(), home_q):
            p.resetJointState(robot.robot_id, j, q)
            p.setJointMotorControl2(robot.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=q)
        block_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), 'cube_small.urdf'),
            [block_xy[0], block_xy[1], TABLE_TOP_Z])
        marker = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), 'cube_small.urdf'),
            [goal_xy[0], goal_xy[1], TABLE_TOP_Z], useFixedBase=True,
            globalScaling=1.6)
        p.changeVisualShape(marker, -1, rgbaColor=[0, 0.8, 0, 0.35])
        p.setCollisionFilterGroupMask(marker, -1, 0, 0)  # marker: visual only
        self_step = int(0.5 / SIM_TIMESTEP)
        for _ in range(self_step):
            p.stepSimulation()

        ok, state, error = fsm.run_episode(block_id, block_xy, goal_xy)
        if ok:
            successes += 1
            print('Episode {:2d}/10 | SUCCESS  (place error {:.3f} m)'.format(
                ep + 1, fsm.place_error))
        else:
            print('Episode {:2d}/10 | FAILURE at {} — {}'.format(
                ep + 1, state, error))

        fsm.log = []
        p.removeBody(block_id)
        p.removeBody(marker)

    score = TASK4_SCORE_MAX * successes / NUM_EPISODES
    print('\n===================================================='
          '================================')
    print('- Your Total Score : {:.3f} / {:.3f}'.format(score, TASK4_SCORE_MAX))
    print('====================================================='
          '===============================')

    # Task 1 bridge: validate the recorded motions with YOUR grounding
    # functions + YOUR shapes.ttl (fail-soft; the score above is final).
    try:
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'semantic'))
        from diagnose import diagnose
        diagnose('Task 4 FSM', fk_records=fsm.diag_fk, ik_records=fsm.diag_ik)
    except Exception as exc:
        print('(semantic diagnosis unavailable: {})'.format(exc))
    p.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', '-g', action='store_true', default=False,
                        help='show the PyBullet window')
    args = parser.parse_args()
    main(args)
