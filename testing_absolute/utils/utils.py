"""Utilities for SfM model creation, correspondence handling, and camera intrinsics."""
import sys
import os
from pathlib import Path
from hloc.utils.io import get_matches

import pycolmap
import numpy as np
import h5py

from hloc import (
    extract_features,
    match_features,
    triangulation,
    pairs_from_retrieval,
)

from .geometry_utils import normal_redirect



def create_model_or_retrieve(images, images_query, outputs, outputs_query, sfm_pairs,
                             sfm_pairs_query, sfm_dir, input_sfm_dir, feat="sift"):
    """Create SfM model via feature extraction/matching or retrieve existing model.
    
    Pipeline: Extract retrieval features -> Match retrieval -> Extract/match local
    features -> Triangulate SfM model -> Create query correspondences.
    
    Args:
        images: Path to reference images directory
        images_query: Path to query images directory
        outputs: Output directory for reference features/matches
        outputs_query: Output directory for query features/matches
        sfm_pairs: Reference image pairs file
        sfm_pairs_query: Query-reference image pairs file
        sfm_dir: Directory for SfM model (created if not exists)
        input_sfm_dir: Initial SfM model from COLMAP
        feat: Feature type ("sift", "sp+lg", "hardaffnet")
    
    Returns:
        model: Reconstructed COLMAP model
        feature_path: Path to reference features (h5)
        feature_query_path: Path to query features (h5)
        match_query_path: Path to query-reference matches (h5)
    """
    # Select feature extraction and matching configuration
    retrieval_conf = extract_features.confs["netvlad"]
    if feat == "sift+NN":
        feature_conf = extract_features.confs["sift"]
        matcher_conf = match_features.confs["NN-ratio"]
    elif feat == "hardaffnet":
        feature_conf = extract_features.confs["hardnet+affnet"]
    elif feat == "sp+lg":
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superpoint+lightglue"]
    else:
        raise ValueError(f"Unknown feature type: {feat}")
    
    # Reference database: retrieve pairs -> extract/match features -> triangulate
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    # Create or load SfM model
    if not os.path.exists(sfm_dir):
        model = triangulation.main(sfm_dir, input_sfm_dir, images, sfm_pairs,
                                   feature_path, match_path, verbose=False)
    else:
        model = get_model(outputs / feat)

    # Query: retrieve pairs -> extract features -> match against reference
    retrieval_path_query = extract_features.main(retrieval_conf, images_query, outputs_query)
    pairs_from_retrieval.main(retrieval_path_query, sfm_pairs_query,
                              db_descriptors=retrieval_path, num_matched=5)
    feature_query_path = extract_features.main(feature_conf, images_query, outputs_query)
    match_query_path = match_features.main(
        matcher_conf, sfm_pairs_query, feature_conf["output"], outputs_query,
        features_ref=feature_path
    )

    return model, feature_path, feature_query_path, match_query_path


def create_corrs(db_image, q_image, feat_path, feat_q_path, match_q_path, kpts_query,
                  model, Ks=None, model_normals_dict=None, return_oriscale=False):
    """Create correspondences between query and reference images.
    
    Matches features between query and database images, retrieves 3D points,
    and optionally computes normals and affine transformations.
    
    Args:
        db_image: Reference image object from COLMAP
        q_image: Query image name
        feat_path: Path to reference features (h5)
        feat_q_path: Path to query features (h5)
        match_q_path: Path to query-reference matches (h5)
        kpts_query: Query keypoints array
        model: COLMAP reconstruction
        Ks: Camera intrinsics [K_ref, K_query] (optional)
        model_normals_dict: Dictionary of 3D point normals (optional)
        return_oriscale: If True, extract orientation/scale from features
    
    Returns:
        If model_normals_dict is None:
            corrs: (N, 5) array [2D query point (2) | 3D world point (3)]
        Else:
            corrs, normals, lafs: Correspondences, surface normals, affine transforms
    """
    # Get matches sorted by SNN ratio
    q_db_matches, score = get_matches(match_q_path, q_image, db_image.name)
    sorted_indices = np.argsort(score)
    q_db_matches = np.array(q_db_matches)[sorted_indices]
    
    # Get 3D point IDs for reference image
    db_image_id = model.find_image_with_name(db_image.name).image_id
    points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                             for p in model.images[db_image_id].points2D])
    
    if points3D_ids.size == 0:  # No 3D points
        if model_normals_dict is None:
            return np.array([0, 5])
        else:
            return np.array([0, 5]), np.array([0, 2]), np.array([0, 2, 2])

    # Collect valid correspondences (those with 3D points)
    mkpq_idx, mkpdb_idx, m3d_idx = [], [], []
    for q_idx, db_m in q_db_matches:
        if points3D_ids[db_m] == -1:
            continue
        mkpq_idx.append(q_idx)
        mkpdb_idx.append(db_m)
        m3d_idx.append(points3D_ids[db_m])

    # Build correspondence array [2D query | 3D world]
    points2D = kpts_query[mkpq_idx].reshape([-1, 2])
    points3D = np.array([model.points3D[j].xyz for j in m3d_idx]).reshape([-1, 3])
    corrs = np.hstack([points2D, points3D])
    
    if model_normals_dict is None:
        return corrs
    
    # Extract focal lengths from intrinsics or use default
    if Ks is None:
        f_r, f_q = 1, 1
    else:
        f_r, f_q = Ks[0][:2, :2].trace() / 2, Ks[1][:2, :2].trace() / 2
        if (abs(Ks[0][0, 0] - Ks[0][1, 1]) > 1.0 or
            abs(Ks[1][0, 0] - Ks[1][1, 1]) > 1.0):
            print("WARNING: Non-square pixels detected!")

    # Extract feature-level information (orientation/scale or affine matrix)
    if return_oriscale:
        # Use SIFT orientation and scale
        lafs_r_data = get_OriScale(feat_path, db_image.name)[mkpdb_idx]
        lafs_q_data = get_OriScale(feat_q_path, q_image)[mkpq_idx]
        
        ori_r, scale_r = lafs_r_data[:, 0], lafs_r_data[:, 1]
        ori_q, scale_q = lafs_q_data[:, 0], lafs_q_data[:, 1]
        
        ori_r_rad = ori_r * np.pi / 180.0  # Convert to radians
        ori_q_rad = ori_q * np.pi / 180.0
        scale_r_norm = scale_r / f_r  # Normalize by focal length
        scale_q_norm = scale_q / f_q
        
        if ori_r_rad.size == 0 or ori_q_rad.size == 0:
            print("WARNING: No valid matches with orientation/scale. Returning empty arrays.")
            return corrs, np.array([0, 2]), np.array([0, 2, 2])
        
        # Validate orientation range
        assert (ori_q_rad.min() >= -np.pi - 0.01 and
                ori_q_rad.max() <= np.pi + 0.01), \
                f"Orientation range error: [{ori_q_rad.min()}, {ori_q_rad.max()}]"
        
        # Stack: [ori_ref, scale_ref, ori_query, scale_query]
        lafs = np.hstack([ori_r_rad.reshape([-1, 1]), scale_r_norm.reshape([-1, 1]),
                          ori_q_rad.reshape([-1, 1]), scale_q_norm.reshape([-1, 1])])
    else:
        # Use affine matrices (LAFs: Local Affine Frames)
        lafs_r = get_AffMat(feat_path, db_image.name)[mkpdb_idx] / f_r
        lafs_q = get_AffMat(feat_q_path, q_image)[mkpq_idx] / f_q
        # Solve for relative affine: lafs = inv(lafs_r) @ lafs_q
        lafs = np.array([np.linalg.solve(lr.T, lq.T).T
                         for lr, lq in zip(lafs_r, lafs_q)])
    
    # Get surface normals for 3D points and orient them towards camera
    normals = np.array([model_normals_dict[m3idx] for m3idx in m3d_idx])
    points3d = np.array([model.points3D[m3idx].xyz for m3idx in m3d_idx])
    viewpoints = np.array([db_image.projection_center() for _ in points3d])
    normals = normal_redirect(points3d, normals, view_point=viewpoints)

    return corrs, normals, lafs


def get_localization_info(file, prefix=""):
    """Parse COLMAP images.txt file to extract poses and camera indices.
    
    Args:
        file: Path to COLMAP images.txt file
        prefix: Prefix to add to image names (optional)
        convert: If True, convert from COLMAP to robot/camera convention
    
    Returns:
        images: List of image names
        cameras: List of camera indices for each image
        poses: Dictionary mapping image names to poses (R, t)
    """
    images, cameras, poses = [], [], {}
    with open(str(file), 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0 or line[0] == "#":
                continue
            data = line.strip().split()
            if len(data) > 16:  # Skip malformed lines
                continue
            # COLMAP format: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name
            pose = np.array(list(map(float, data[1:8])))
            images.append(data[-1])
            cameras.append(int(data[-2]))
            # Extract quaternion (wxyz) and translation
            poses[prefix + data[9]] = pycolmap.Rigid3d(pycolmap.Rotation3d(pose[:4]), pose[4:])
    return images, cameras, poses


def get_intrinsics(file):
    """Parse COLMAP cameras.txt file to extract camera intrinsics."""
    intrinsics = {}
    with open(str(file), 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0 or line[0] == "#":
                continue
            data = line.strip().split()
            camera, model, width, height, *params = data
            params = np.array(params, float)
            intrinsics[int(camera)] = pycolmap.Camera(
                camera_id=int(camera), model=model,
                width=int(width), height=int(height), params=params)
    return intrinsics

def calculate_error(gt_pose, est_pose):
    """Compute rotation and translation errors between ground truth and estimate."""
    R2R1 = np.dot(gt_pose[:, 0:3].T, est_pose[:, 0:3])
    cos_angle = max(-1.0, min(1.0, 0.5 * (R2R1.trace() - 1.0)))
    
    err_R = np.arccos(cos_angle) * 180.0 / np.pi
    err_t = np.linalg.norm(-gt_pose[:, 0:3].T @ gt_pose[:, 3] -
                           (-est_pose[:, 0:3].T @ est_pose[:, 3]))
    
    return err_R, err_t


def get_model(sfm_path: Path) -> pycolmap.Reconstruction:
    """Load COLMAP reconstruction from directory."""
    model = pycolmap.Reconstruction(sfm_path)
    print("Got model:\n\t", model)
    return model

def get_AffMat(path: Path, name: str) -> np.ndarray:
    """Load affine matrices (Local Affine Frames) from h5 file."""
    with h5py.File(str(path).replace(".h5", "_lafs.h5"), "r", libver="latest") as hfile:
        lafs = hfile[name]["lafs"].__array__()
    return lafs.astype(np.float32)


def get_OriScale(path: Path, name: str) -> np.ndarray:
    """Load SIFT orientation and scale from h5 file."""
    with h5py.File(str(path).replace(".h5", "_lafs.h5"), "r", libver="latest") as hfile:
        oris = hfile[name]["oris"].__array__()  # In degrees
        scales = hfile[name]["scales"].__array__()
    return np.hstack([oris.reshape([-1, 1]), scales.reshape([-1, 1])]).astype(np.float32)

def compute_rotation_to_axis(R_ref, observation, target_axis=np.array([0, 1, 0])):
    """Compute rotation matrix to align observation vector with target axis.
    
    Uses Rodrigues' rotation formula to find the minimum rotation from g_cam to target.
    Handles special cases: already aligned (identity) and opposite direction (180°).
    
    Args:
        R_ref: Reference rotation matrix
        observation: Vector to rotate (will be normalized)
        target_axis: Desired direction (default: [0, 1, 0])
    
    Returns:
        (3, 3) rotation matrix
    """
    # Apply reference rotation to observation
    g_cam = R_ref @ observation
    g_cam = g_cam / np.linalg.norm(g_cam)  # Normalize
    
    # Ensure target is unit vector
    target = target_axis / np.linalg.norm(target_axis)
    
    # Compute rotation axis and angle via cross/dot product
    v = np.cross(g_cam, target)
    c = np.dot(g_cam, target)  # cos(theta)
    s = np.linalg.norm(v)      # sin(theta)
    
    # Handle special cases to avoid numerical issues
    if s < 1e-8:
        if c > 0:  # Already aligned
            return np.eye(3)
        else:      # Exactly opposite (180 degrees)
            # Find arbitrary orthogonal axis
            ortho = np.array([1, 0, 0]) if abs(g_cam[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(g_cam, ortho)
            axis /= np.linalg.norm(axis)
            # 180 degree rotation: R = -I + 2*nn^T
            return -np.eye(3) + 2 * np.outer(axis, axis)
    
    # Rodrigues' formula: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
    skew = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    Rxz = np.eye(3) + skew + (skew @ skew) * ((1 - c) / (s ** 2))
    return Rxz
