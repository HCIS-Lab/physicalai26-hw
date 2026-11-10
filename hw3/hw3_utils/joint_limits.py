"""Single source of truth for the course's UR5 joint limits (rad).

Used by ik.py (your_ik and the pybullet_ik reference) and semantic/common.py.
"""
import numpy as np

UR5_JOINT_LIMITS = np.asarray([
    [-3 * np.pi / 2, -np.pi / 2],   # joint1
    [-2.3562, -1.0],                # joint2
    [-17.0, 17.0],                  # joint3
    [-17.0, 17.0],                  # joint4
    [-17.0, 17.0],                  # joint5
    [-17.0, 17.0],                  # joint6
])
