"""Solver definitions for absolute pose estimation solvers."""
from enum import Enum

class Solvers(Enum):
    """Enumeration of available solvers forAbsolute Pose Estimation:
    - COLMAP:     Reference COLMAP solver P3P
    - PNP_RANSAC: OpenCV's solvePnPRansac
    - P3P:        GC-RANSAC 3-point pose solver
    - UP2P_ERI:   Gravity-aware 2-point solver with rotation invariance
    - UP1P_SIFT:  Gravity-aware 1-point with SIFT scale/orientation
    - P1P_AC:     1-point affine features solver
    - P4PF:       Uncalibrated 4-point focal length solver
    - P35PF:      Uncalibrated 3.5-point focal length solver
    - UP3PF:      Uncalibrated gravity-aware 3-point 
    - UP1PF_AC:   Uncalibrated gravity-aware 1-point and affine features
    - UP2PFORI:   Uncalibrated gravity-aware 2-point and orientation
    """
    COLMAP = 0
    PNP_RANSAC = 1
    P3P = 2
    UP2P_ERI = 3
    UP1P_SIFT = 4
    P1P_AC = 5
    P4PF = 6
    P35PF = 7
    UP3PF = 8
    UP1PF_AC = 9
    UP2PFORI = 10
