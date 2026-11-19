# -*- coding: utf-8 -*-
"""Step-by-step forward-kinematics trace (TA-provided, no student TODO).

Forward kinematics is a CHAIN of 4x4 matrix multiplications:

    A_0 = base transform
    A_k = A_(k-1) @ T_k        (k = 1..6,  T_k = local DH transform of joint k)

trace_fk() runs this chain and returns every T_k and every cumulative A_k,
so each multiplication step can be written to the knowledge graph as an
hw3:FKStep and later checked with SHACL / SPARQL (Task 4).

The multiplication is pluggable: `matmul(prev, local, k)` returns A_k.
 - the CORRECT one is matmul_correct (prev @ local) -> output/Student.ttl
 - the TA's execution (semantic/TA.ttl) was produced by a program whose
   multiplication is WRONG at one step; Task 4 asks you to find where and why.
"""

import numpy as np

from hw3_utils.bullet_utils import get_matrix_from_pose


def dh_transform(dh, theta):
    """Classic DH transform  T = Rz(theta) Tz(d) Tx(a) Rx(alpha)  (4x4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(dh['alpha']), np.sin(dh['alpha'])
    return np.array([
        [ct, -st * ca,  st * sa, dh['a'] * ct],
        [st,  ct * ca, -ct * sa, dh['a'] * st],
        [0.,       sa,       ca,      dh['d']],
        [0.,       0.,       0.,           1.],
    ])


def matmul_correct(prev, local, k):
    """A_k = A_(k-1) @ T_k"""
    return prev @ local


def trace_fk(dh_params, q, base_pos, matmul=matmul_correct):
    """Return [{'index': k, 'local': T_k, 'cumulative': A_k}, ...] for k = 0..6.

    k = 0 is the base: local = identity, cumulative = base transform.
    """
    base = np.asarray(get_matrix_from_pose(list(base_pos) + [0, 0, 0]),
                      dtype=float)
    steps = [{'index': 0, 'local': np.eye(4), 'cumulative': base}]
    A = base
    for k, (dh, theta) in enumerate(zip(dh_params, q), start=1):
        T = dh_transform(dh, theta)
        A = matmul(A, T, k)
        steps.append({'index': k, 'local': T, 'cumulative': A})
    return steps
