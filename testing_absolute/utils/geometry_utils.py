"""Geometry utilities for coordinate transformations and normal estimation."""
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d


def rot(x, axis):
    """Quick rotation matrix from Euler angles."""
    return Rotation.from_euler(axis, x).as_matrix()

def to_o3d_pcd(pcd):
    """Convert numpy point cloud to Open3D PointCloud.
    
    Args:
        pcd: Point cloud as numpy.ndarray of shape (N, 3)
    
    Returns:
        open3d.geometry.PointCloud object
    """
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_


def estimate_normals(model, neighbors=200):
    """Estimate surface normals for 3D model points using Open3D.
    
    Args:
        model: COLMAP reconstruction object
        neighbors: Number of neighbors for normal estimation (default: 200)
    
    Returns:
        Dictionary mapping point3D ids to their surface normals
    """
    p3d = np.array([pt.xyz for pt in model.points3D.values()])
    o3dmodel = to_o3d_pcd(p3d)
    o3dmodel.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Reset normals
    o3dmodel.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=neighbors))
    o3dmodel.orient_normals_consistent_tangent_plane(k=15)  # Orient consistently
    o3dmodel_normals = np.asarray(o3dmodel.normals)
    
    model_normals_dict = {pt: o3dmodel_normals[i] for i, pt in enumerate(model.points3D.keys())}
    return model_normals_dict


def normalize_image_points(corrs, K):
    """Normalize 2D image points using camera intrinsics.
    
    Args:
        corrs: (N, 5) array [2D points (2) | 3D points (3)]
        K: Camera intrinsic matrix (3, 3)
    
    Returns:
        normalized_correspondences: (N, 5) with normalized 2D coordinates
    """
    n = len(corrs)
    normalized_correspondences = np.zeros((corrs.shape[0], 5))
    inv_K = np.linalg.inv(K)

    for i in range(n):
        p1 = np.append(corrs[i][0:2], 1)
        p2 = inv_K.dot(p1)  # Apply inverse intrinsics
        normalized_correspondences[i][0:2] = p2[0:2]
        normalized_correspondences[i][2:] = corrs[i][2:]
    return normalized_correspondences

def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point(s)
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals
    return rot