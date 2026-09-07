"""
Geometry-only ICP SLAM utilities for HW1 — student distribution.

STUDENT IMPLEMENTATION SURFACE:
    1. depth_image_to_point_cloud: Back-project RGB-D frames into 3D PointClouds.
    2. my_local_icp_algorithm: From-scratch point-to-plane/point-to-point ICP with SVD.
"""

import json
import numpy as np
import open3d as o3d
import os
import time
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DEPTH_SCALE = 1000.0
CAM_TO_GT_AXES = np.diag([1.0, -1.0, -1.0])


def load_depth_meters(depth_path):
    """Read a depth PNG from disk and return it as a float64 depth map in metres."""
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.dtype == np.uint16:
        return d.astype(np.float64) / DEPTH_SCALE
    return d.astype(np.float64) / 255.0 * 10.0


# =============================================================================
# TODO: Student Assignment Task 1 (3D Back-projection)
# =============================================================================
def depth_image_to_point_cloud(rgb, depth_m, width, height, hfov, keep_mask=None):
    """
    Back-project one RGB-D frame into a colored 3D point cloud.

    CONTRACT:
        Inputs:
            rgb      : H*W*3 uint8, channels in BGR order (from cv2.imread).
            depth_m  : H*W float array, depth in METRES.
            width    : int, sensor width in pixels.
            height   : int, sensor height in pixels.
            hfov     : float, horizontal field of view in DEGREES.
            keep_mask: optional H*W boolean mask; drop pixels where keep_mask is False.

        Output:
            o3d.geometry.PointCloud carrying:
              .points : (N, 3) float64 in camera coordinates (+X right, +Y down, +Z forward).
              .colors : (N, 3) float in [0, 1], in RGB order.

        Validity:
            A pixel contributes a point iff depth_m > 0 (and keep_mask is True if provided).
            Zero and negative depths must be excluded.

        Note:
            Open3D projection helpers are strictly forbidden. Use pinhole camera math.
    """
    h, w = depth_m.shape
    if (height, width) != (h, w):
        raise ValueError(
            f"intrinsics ({width}x{height}) do not match frame ({w}x{h})")

    # TODO: Student implementation
    # 1. Compute focal lengths fx, fy and optical centre cx, cy using width, height, and hfov
    # 2. Filter valid pixels where depth_m > 0 (and apply keep_mask if provided)
    # 3. Unproject image coordinates (u, v, Z) to camera coordinates (X, Y, Z):
    #       X = (u - cx) * Z / fx
    #       Y = (v - cy) * Z / fy
    # 4. Construct and return an o3d.geometry.PointCloud with points and normalized RGB colors
    raise NotImplementedError("TODO: Implement depth_image_to_point_cloud")


def preprocess_point_cloud(pcd, voxel_size):
    """Voxel-downsample cloud and estimate normals/FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
    return pcd_down, fpfh


def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Estimate initial alignment using RANSAC with FPFH descriptors."""
    dist_thr = voxel_size * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """Refine alignment with Open3D Point-to-Plane ICP."""
    for pcd in (source_down, target_down):
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))

    return o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))


def multiscale_icp(source_down, target_down, trans_init,
                   thresholds=(0.4, 0.2, 0.1, 0.05), max_iter=60):
    """Multi-scale point-to-plane ICP refinement."""
    for pcd in (source_down, target_down):
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    T = trans_init
    for thr in thresholds:
        T = o3d.pipelines.registration.registration_icp(
            source_down, target_down, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)).transformation

    class _Result:
        def __init__(self, transformation):
            self.transformation = transformation
    return _Result(T)


# =============================================================================
# TODO: Student Assignment Task 2 (From-Scratch SVD ICP)
# =============================================================================
def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Custom ICP implementation from scratch using SVD.

    CONTRACT:
        Inputs:
            source_down: o3d.geometry.PointCloud, source point cloud.
            target_down: o3d.geometry.PointCloud, target point cloud.
            trans_init : 4x4 float ndarray, initial relative transformation estimate.
            voxel_size : float, voxel grid resolution in metres.

        Returns:
            An object with a `.transformation` attribute containing the final 4x4 transform.

        Specification:
            - Set max_correspondence_distance = voxel_size * 1.5, max_iter = 60, tol = 1e-6.
            - Iterative procedure:
                1. Transform source points by current transform T.
                2. Find closest points in target using KD-tree (e.g., scipy.spatial.cKDTree).
                3. Reject correspondence pairs with Euclidean distance >= threshold.
                4. Estimate rigid transform (R, t) minimizing error using SVD (Kabsch/Umeyama).
                5. Check determinant of R to prevent reflection (ensure det(R) == +1).
                6. Update cumulative transform and check convergence.
    """
    # TODO: Student implementation
    # 1. Setup threshold, iteration limits, and KD-Tree over target points
    # 2. Loop through iterations: query correspondences, filter outliers
    # 3. Compute cross-covariance matrix H = (P - p_mean)^T @ (Q - q_mean)
    # 4. Perform SVD and derive optimal R and t
    # 5. Return duck-typed object exposing .transformation
    raise NotImplementedError("TODO: Implement my_local_icp_algorithm")

# =============================================================================
# [REST OF utils.py (RECONSTRUCT PIPELINE, MEAN_L2, VISUALIZATION) REMAINS UNMODIFIED]
# =============================================================================