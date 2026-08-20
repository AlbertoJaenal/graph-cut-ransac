"""Evaluation utilities for localization and pose estimation."""
import numpy as np

from .utils import get_intrinsics, get_localization_info
from pathlib import Path

from hloc.utils.read_write_model import qvec2rotmat


def evaluate_generic(images_query, results, only_localized=False):
    """Evaluate pose estimation results against ground truth.
    
    Computes median translation, rotation, focal length errors and execution time.
    
    Args:
        images_query: Path to query images list
        results: Path to results file with estimated poses and focal lengths
        only_localized: If True, only evaluate successfully localized images
    
    Prints evaluation metrics: cm, degrees, focal error, time (ms), localization count
    """
    # Parse prediction results
    predictions = {}
    with open(results, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:-2], float), [4])  # Quaternion and translation
            f, tim = np.array(data[-2:], float)  # Focal length and time
            predictions[name] = (qvec2rotmat(q), t, f, tim)
    # Load intrinsics and poses
    query_cameras = get_intrinsics(str(images_query).replace("images", "cameras") + ".txt")
    query_ims, qim_cameras, qim_poses = get_localization_info(
        str(images_query) + ".txt")

    # Initialize error lists
    errors_t = []
    errors_R = []
    errors_f = []
    tims = []
    N = 0  # Count of successfully localized images
    for i, name in enumerate(query_ims):
        if name not in predictions:
            # No prediction for this image
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            # Compare estimated pose with ground truth
            R, t, f, tim = predictions[name]
            
            # Count valid localization (non-zero translation)
            if not np.allclose(t, np.zeros([3])) and not np.any(np.isinf(t)) and not np.any(np.isnan(t)):
                N += 1
            
            # Get ground truth focal length
            f_gt = query_cameras[qim_cameras[i]].calibration_matrix()[:2, :2].trace() / 2
            
            # COLMAP format
            R_gt, t_gt = qim_poses[name].rotation.matrix(), qim_poses[name].translation
            
            # Compute translation error (in meters)
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            
            # Compute rotation error (in degrees)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            
            # Compute focal length error (relative)
            e_f = np.abs(f_gt - f) / f_gt
            
        errors_t.append(e_t)
        errors_R.append(e_R)
        errors_f.append(e_f)
        tims.append(tim)

    # Convert to numpy arrays and compute medians
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    errors_f = np.array(errors_f)
    tims = np.array(tims)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    med_f = np.median(errors_f)
    med_tim = np.median(tims) * 1e3  # Convert to milliseconds
    
    # Header for output
    out = "cm,deg,f,tim,loc,total"
    result = ""
    result += f"\n{med_t*100:.3f},{med_R:.3f},{med_f:.3f},{med_tim:.3f},{N},{len(query_ims)}"

    # Compute localization ratios at specific thresholds
    threshs_t = [0.05, 0.1]  # Meters
    threshs_R = [1.0, 1.0]    # Degrees
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f",{th_t*100:.0f}cm-{th_R:.0f}deg"
        result += f",{ratio*100:.2f}"
    
    print(out, result)

def evaluate_ftim(data_dir, result_fil, do_print=False):
    """Evaluate focal length and timing predictions for Aachen dataset."""
    # Load focal length ground truth
    focals, predictions = {}, {}
    for i in ["day", "night"]:
        with open(Path(data_dir) / f"queries/{i}_time_queries_with_intrinsics.txt", "r") as f:
            for data in f.read().rstrip().split("\n"):
                data = data.split()
                name = data[0]
                f = float(data[4])
                focals[name.split("/")[-1]] = f

    # Parse predictions
    with open(result_fil, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            f, tim = np.array(data[-2:], float)
            predictions[name] = (f, tim)

    # Compute median errors
    errors_f = []
    tims = []
    for name in focals.keys():
        if name not in predictions:
            raise ValueError(f"Missing prediction for {name}")
        f, tim = predictions[name]
        e_f = np.abs(focals[name] - f) / focals[name]
        errors_f.append(e_f)
        tims.append(tim)

    errors_f = np.array(errors_f)
    tims = np.array(tims)

    med_f = np.median(errors_f)
    med_tim = np.median(tims) * 1e3  # Convert to milliseconds
    out = f"f,tim\n{med_f:.3f},{med_tim:.3f}\n"
    
    if do_print:
        print(out)
    # Save to file
    with open(str(result_fil).replace(".txt", "_ftim.txt"), "w") as f:
        f.write(out)
