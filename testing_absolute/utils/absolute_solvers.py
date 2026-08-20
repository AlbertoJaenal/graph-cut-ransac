"""Pose verification solvers using OpenCV and GC-RANSAC backends."""
import cv2
import numpy as np

from . import pygcransac


def get_probabilities(tentatives):
    """Compute inlier probabilities based on SNN ratio ordering (closer to top = higher probability)."""
    probabilities = []
    for i in range(len(tentatives)):
        probabilities.append(1.0 - i / len(tentatives))
    return probabilities

def estimate_abspos_opencv(normalized_corrs, K):
    """Estimate absolute pose using OpenCV's solvePnPRansac.

    Args:
        normalized_corrs: Normalized correspondence data (Nx5 array: [u, v, X, Y, Z])
        K: Camera intrinsic matrix

    Returns:
        pose: (3, 4) camera pose [R|t]
        mask: (N,) binary inlier mask
    """
    n = len(normalized_corrs)
    imagePoints = np.float32([normalized_corrs[i][0:2] for i in np.arange(n)]).reshape(-1,2)
    worldPoints = np.float32([normalized_corrs[i][2:5] for i in np.arange(n)]).reshape(-1,3)
    dist_coeffs = np.zeros((4,1))
    camera_matrix = np.identity(3)

    threshold = 2.0
    normalized_threshold = threshold / (K[0, 0] + K[1, 1]) / 2.0;    

    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        worldPoints, 
        imagePoints, 
        camera_matrix, 
        dist_coeffs, 
        flags = cv2.SOLVEPNP_ITERATIVE,
        iterationsCount = 1000,
        reprojectionError = normalized_threshold)
    
    mask = np.zeros(n)
    if not success:
        return np.zeros((3, 4)), mask
    mask[inliers] = 1
    
    rotation, _ = cv2.Rodrigues(rotation_vector)
    pose = np.concatenate((rotation, translation_vector), axis=1)
    return pose, mask

def estimate_abspos_pygcransac(corrs, solver=0, sampler_id=0, normalized_threshold=2.0):
    """Estimate absolute pose using GC-RANSAC (6-DOF).
    
    Args:
        corrs: Correspondence data (format depends on solver)
        solver: Solver index (0: P4PF, 1: P3.5PF, 2: UP3PF, 3: UP1PF_AC, 4: UP2PFORI)
        sampler_id: Sampler strategy ID
        normalized_threshold: Inlier threshold for pose estimation normalized by query focal length
    
    Returns:
        pose: (3, 4) camera pose [R|t]
        mask: (N,) binary inlier mask
        num_inl: Number of inliers
    """
    pose, mask, num_inl = pygcransac.find6DPose(
        np.ascontiguousarray(corrs),
        min_iters = 50,
        max_iters = 5000,
        probabilities = [], # Inlier probabilities. This is not used if the sampler is not 3 (NG-RANSAC) or 4 (AR-Sampler)
        sampler = sampler_id, # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)
        threshold = normalized_threshold,  # Inlier-outlier threshold
        conf = 0.99, # RANSAC confidence
        solver = solver) 
    return pose, mask, num_inl

def estimate_abspos_sift_pygcransac(corrs, solver=2, sampler_id=0, normalized_threshold=2.0):
    """Estimate absolute pose using GC-RANSAC with SIFT features (6-DOF).

    Args:
        corrs: Correspondence data (format depends on solver)
        solver: Solver index (0: P4PF, 1: P3.5PF, 2: UP3PF, 3: UP1PF_AC, 4: UP2PFORI)
        sampler_id: Sampler strategy ID
        normalized_threshold: Inlier threshold for pose estimation normalized by query focal length
    
    Returns:
        pose: (3, 4) camera pose [R|t]
        mask: (N,) binary inlier mask
        num_inl: Number of inliers
    """
    inlier_probabilities = []
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(corrs)
    
    pose, mask, num_inl = pygcransac.find6DPoseSIFT(
        np.ascontiguousarray(corrs),
        max_iters = 5000, # The maximum number of iterations
        min_iters = 50, # The minimum number of iterations
        probabilities = inlier_probabilities, # The inlier probabilities for all points
        sampler = sampler_id, # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)
        threshold = normalized_threshold,  # Inlier-outlier threshold
        spatial_coherence_weight = 0.4,
        min_inlier_ratio_for_sprt= 0.01,
        neighborhood_size=20,
        solver = solver)
    
    return pose, mask, num_inl

def estimate_abspos_f_pygcransac(corrs, solver=0, threshold=4.0, max_iters=5000, min_iters=50, sampler_id=0, lo_number=50):
    """Estimate Uncalibrated Absolute Pose using GC-RANSAC with or without extra info e.g. SIFT features (7-dof).
    
    Args:
        corrs: Correspondence data (format depends on solver)
        solver: Solver index (0: P4PF, 1: P3.5PF, 2: UP3PF, 3: UP1PF_AC, 4: UP2PFORI)
        threshold: Inlier threshold for pose estimation
        max_iters: Maximum RANSAC iterations
        min_iters: Minimum RANSAC iterations
        sampler_id: Sampler strategy ID
        lo_number: Number of local optimization iterations
    
    Returns:
        pose: (3, 4) camera pose [R|t]
        f: Estimated focal length
        mask: (N,) binary inlier mask
        num_inl: Number of inliers
    """
    inlier_probabilities = []
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(corrs)

    pose, f, mask, num_inl = pygcransac.find6DPoseF(
        np.ascontiguousarray(corrs).astype(np.float64),
        max_iters = max_iters,
        min_iters = min_iters,
        probabilities = np.array(inlier_probabilities), # Inlier probabilities. This is not used if the sampler is not 3 (NG-RANSAC) or 4 (AR-Sampler)
        sampler = sampler_id, # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)
        threshold = threshold,  # Inlier-outlier threshold
        spatial_coherence_weight = 0.4,
        min_inlier_ratio_for_sprt= 0.01,
        neighborhood_size=20,
        lo_number=lo_number,
        sampler_variance=0.1,
        solver = solver) 
    return pose, f, mask, num_inl