"""Absolute pose estimation using GC-RANSAC solvers."""
import pycolmap
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Union

from hloc.localize_sfm import QueryLocalizer
from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.io import get_keypoints
from hloc import logger

from .absolute_defs import Solvers
from .geometry_utils import estimate_normals, normalize_image_points
from .utils import create_corrs, compute_rotation_to_axis
from .absolute_solvers import (
    estimate_abspos_opencv,
    estimate_abspos_pygcransac,
    estimate_abspos_sift_pygcransac,
    estimate_abspos_f_pygcransac
)

from multiprocessing import Pool
import multiprocessing as mp


def pose_from_gcransac(
    localizer: QueryLocalizer,
    qname: str,
    query_camera: pycolmap.Camera,
    db_ids: List[int],
    features_path: Path,
    matches_path: Path,
    solver: Solvers,
    features_query_path: Path = None,
    model_normals_dict: Dict = None,
    gravity_vector_world=[0, 1, 0],
    **kwargs,
):
    """Estimate absolute pose for a query image using GC-RANSAC solvers.
    
    Iterates over retrieved database images, creates correspondences, and runs
    the specified solver to estimate camera pose and optionally focal length.
    
    Returns:
        ret: Dictionary with 'cam_from_world' (pose), 'camera' (focal), 'time'
        log: Dictionary with solver name and metadata
    """
    ###################
    # Preparation
    ###################
    if features_query_path is None:
        features_query_path = features_path

    kpq = get_keypoints(features_query_path, qname)
    kpq += 0.5  # Convert to COLMAP coordinates
    K_query = query_camera.calibration_matrix()

    # Initialize tracking variables for best pose
    inlier_num, t1, T_w2q, F = 0, 10, np.eye(4), 0
    NAME = ""

    for i, db_id in enumerate(db_ids):
        # Get reference image and its camera intrinsics
        image = localizer.reconstruction.images[db_id]
        K_ref = image.camera.calibration_matrix()
        f_ref = (K_ref[:2, :2].trace() / 2).reshape([1, 1])

        # Get reference image pose in world frame
        T_w2ref = np.eye(4)
        T_w2ref[:3] = image.cam_from_world().matrix()

        normals, lafs = None, None
        Rxz = np.eye(3)
        
        # Compute gravity alignment rotation for upright solvers
        if solver in [Solvers.UP2P_ERI, Solvers.UP1P_SIFT, Solvers.UP3PF,
                      Solvers.UP1PF_AC, Solvers.UP2PFORI]:
            g_ref = T_w2ref[:3, :3] @ gravity_vector_world
            g_ref = g_ref / np.linalg.norm(g_ref)
            g_query_approx = np.array([0, -1, 0])  # Assume upright query camera

            Rxz = compute_rotation_to_axis(np.eye(3), g_query_approx, target_axis=g_ref)
        
        # Apply gravity alignment to reference pose
        T_w2ref[:3, :3] = Rxz @ T_w2ref[:3, :3]
        T_w2ref[:3, 3] = Rxz @ T_w2ref[:3, 3]

        # Create correspondence matrix with optional normals/affines for AC solvers
        if solver in [Solvers.P1P_AC, Solvers.UP1P_SIFT, Solvers.UP1PF_AC, Solvers.UP2PFORI]:
            # Determine which intrinsics to use for normalization
            if solver in [Solvers.P1P_AC, Solvers.UP1P_SIFT]:
                Ks = [K_ref, K_query]
            else:
                Ks = [K_ref, K_ref]

            corrs, normals, lafs = create_corrs(
                image, qname, features_path, features_query_path, matches_path,
                kpq, localizer.reconstruction,
                Ks=Ks, model_normals_dict=model_normals_dict,
                return_oriscale=solver in [Solvers.UP1P_SIFT, Solvers.UP2PFORI])
        else:
            corrs = create_corrs(
                image, qname, features_path, features_query_path, matches_path,
                kpq, localizer.reconstruction)
        
        if len(corrs) < 30:
            continue

        # Transform 3D points to reference camera frame
        corrs[:, 2:5] = (T_w2ref[:3, :3] @ corrs[:, 2:5].T).T + T_w2ref[:3, 3]

        # Transform normals if available
        if normals is not None:
            normals = (T_w2ref[:3, :3] @ normals.T).T
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        # Normalize 2D points by camera intrinsics
        if solver in [Solvers.P4PF, Solvers.P35PF, Solvers.UP3PF,
                      Solvers.UP1PF_AC, Solvers.UP2PFORI]:
            # For focal estimation: use common principal point
            K_cprr = np.copy(K_ref)
            K_cprr[:2, 2] = K_query[:2, 2]
            corrs = normalize_image_points(corrs, K_cprr)
        else:
            corrs = normalize_image_points(corrs, K_query)

        ###################
        # Run solver
        ###################
        sampler = 0
        t0 = time.time()
        f_est, solver_info = None, None
        threshold = 2.0
        
        # Select and run appropriate solver
        if solver == Solvers.PNP_RANSAC:
            NAME = "[OPENCV] solvePnPRansac"
            pose, mask = estimate_abspos_opencv(corrs, K_query)
            num_inl = mask.sum()

        elif solver == Solvers.P3P:
            NAME = "[GC-RANSAC] P3P"
            # Get focal length from query camera for threshold normalization
            f_query = (K_query[:2, :2].trace() / 2).reshape([1, 1])
            F = f_query.item()
            normalized_threshold = threshold / f_query.item()

            pose, mask, num_inl = estimate_abspos_pygcransac(
                corrs, sampler_id=sampler, normalized_threshold=normalized_threshold)

        elif solver == Solvers.UP2P_ERI:
            NAME = "[GC-RANSAC] UP2P_eri"
            # Get focal length from query camera for threshold normalization
            f_query = (K_query[:2, :2].trace() / 2).reshape([1, 1])
            F = f_query.item()
            normalized_threshold = threshold / f_query.item()

            solver_info = np.hstack([corrs, np.repeat(Rxz.flatten()[None], len(corrs), 0)])
            pose, mask, num_inl = estimate_abspos_pygcransac(
                solver_info, solver=2, sampler_id=sampler,
                normalized_threshold=normalized_threshold)

        elif solver == Solvers.P1P_AC:
            NAME = "[GC-RANSAC] P1P_AC"
            # Get focal length from query camera for threshold normalization
            f_query = (K_query[:2, :2].trace() / 2).reshape([1, 1])
            F = f_query.item()
            normalized_threshold = threshold / f_query.item()

            solver_info = np.hstack([corrs, normals.reshape([-1, 3]), lafs.reshape([-1, 4])])
            pose, mask, num_inl = estimate_abspos_sift_pygcransac(
                solver_info, solver=2, sampler_id=sampler,
                normalized_threshold=normalized_threshold)

        elif solver == Solvers.UP1P_SIFT:
            NAME = "[GC-RANSAC] UP1P_SIFT"
            # Get focal length from query camera for threshold normalization
            f_query = (K_query[:2, :2].trace() / 2).reshape([1, 1])
            F = f_query.item()
            normalized_threshold = threshold / f_query.item()

            solver_info = np.hstack([corrs,
                                     np.repeat(np.ones([1, 1]), len(corrs), 0),
                                     normals.reshape([-1, 3]),
                                     lafs.reshape([-1, 4]),
                                     np.repeat(Rxz.flatten()[None], len(corrs), 0),
                                     np.repeat(np.eye(3).flatten()[None], len(corrs), 0),
                                     np.repeat(np.zeros([3]).flatten()[None], len(corrs), 0)])
            pose, mask, num_inl = estimate_abspos_sift_pygcransac(
                solver_info, solver=3, sampler_id=sampler,
                normalized_threshold=normalized_threshold)

        elif solver == Solvers.P4PF:
            NAME = "[GC-RANSAC] P4Pf"
            pose, f_est, mask, num_inl = estimate_abspos_f_pygcransac(
                corrs, solver=0, lo_number=30, sampler_id=sampler,
                threshold=40.0 / f_ref.item())

        elif solver == Solvers.P35PF:
            NAME = "[GC-RANSAC] P3.5Pf"
            pose, f_est, mask, num_inl = estimate_abspos_f_pygcransac(
                corrs, solver=1, lo_number=30, sampler_id=sampler,
                threshold=40.0 / f_ref.item())

        elif solver == Solvers.UP3PF:
            NAME = "[GC-RANSAC] UP3Pf"
            solver_info = np.hstack([corrs, np.repeat(Rxz.flatten()[None], len(corrs), 0)])
            pose, f_est, mask, num_inl = estimate_abspos_f_pygcransac(
                solver_info, solver=2, lo_number=30, sampler_id=sampler,
                threshold=40.0 / f_ref.item())

        elif solver == Solvers.UP1PF_AC:
            NAME = "[GC-RANSAC] UP1Pf_AC"
            solver_info = np.hstack([corrs,
                                     np.repeat(np.ones([1, 1]), len(corrs), 0),
                                     normals.reshape([-1, 3]),
                                     lafs.reshape([-1, 4]),
                                     np.repeat(Rxz.flatten()[None], len(corrs), 0),
                                     np.repeat(np.eye(3).flatten()[None], len(corrs), 0),
                                     np.repeat(np.zeros([3]).flatten()[None], len(corrs), 0)])
            pose, f_est, mask, num_inl = estimate_abspos_f_pygcransac(
                solver_info, solver=3, lo_number=30, sampler_id=sampler,
                threshold=15.0 / f_ref.item())

        elif solver == Solvers.UP2PFORI:
            NAME = "[GC-RANSAC] UP2PfOri"
            solver_info = np.hstack([corrs,
                                     np.repeat(np.ones([1, 1]), len(corrs), 0),
                                     normals.reshape([-1, 3]),
                                     lafs.reshape([-1, 4]),
                                     np.repeat(Rxz.flatten()[None], len(corrs), 0),
                                     np.repeat(np.eye(3).flatten()[None], len(corrs), 0),
                                     np.repeat(np.zeros([3]).flatten()[None], len(corrs), 0)])
            pose, f_est, mask, num_inl = estimate_abspos_f_pygcransac(
                solver_info, solver=4, lo_number=30, sampler_id=sampler,
                threshold=15.0 / f_ref.item())

        else:
            raise NotImplementedError
        
        ###################
        # Result extraction
        ###################
        
        # Compute final focal length
        f = f_est * f_ref.squeeze() if f_est is not None else f_query.squeeze()
        t1_ = time.time() - t0

        # Keep best pose across all retrieved images
        if num_inl > inlier_num:
            F, t1, inlier_num = f, t1_, num_inl
            T_ref2q = np.eye(4)
            T_ref2q[:3] = pose
            T_w2q = T_ref2q @ T_w2ref

    del corrs, normals, lafs, kpq

    ret = {
        "cam_from_world": pycolmap.Rigid3d(T_w2q[:3]),
        "camera": F,
        "time": t1
    }
    return ret, {"name": NAME}

def _init_worker(sfm_path, config, model_normals_dict=None):
    """This runs ONCE when each worker process is created."""
    global worker_data
    recon = pycolmap.Reconstruction(sfm_path)
    worker_data['localizer'] = QueryLocalizer(recon, config)
    worker_data['db_name_to_id'] = {img.name: i for i, img in recon.images.items()}
    worker_data['model_normals_dict'] = model_normals_dict

def _process_query_worker(args):
    """Worker function for parallel query processing."""
    global worker_data
    qname, qcam, retrieval_dict, features, matches, solver, \
        features_query_path, gravity_vector_world = args
    
    # Access the pre-loaded data
    localizer = worker_data['localizer']
    db_name_to_id = worker_data['db_name_to_id']
    
    # Prepare data
    if qname not in retrieval_dict:
        raise ValueError(f"No images retrieved for query image {qname}.")
    
    db_names = retrieval_dict[qname]
    db_ids = [db_name_to_id[n] for n in db_names]

    # Get pose using GC-RANSAC
    ret, log = pose_from_gcransac(
            localizer, qname, qcam, db_ids, features, matches, solver=solver, features_query_path=features_query_path,
            model_normals_dict=worker_data['model_normals_dict'], gravity_vector_world=gravity_vector_world
        )
    
    return qname, ret, log

def main_gcloc(
    reference_sfm_path: Union[Path, pycolmap.Reconstruction],
    queries: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    results: Path,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
    solver: Solvers = Solvers.COLMAP,
    features_query_path: Path = None,
    gravity_vector_world = None,
    parallel: int = -1,
):
    """
    Localize query images using GC-RANSAC solvers with a reference SfM model, given a solver and configuration. Supports parallel processing of queries.
    Args:
        reference_sfm_path: Path to the reference SfM model or a pycolmap.Reconstruction object.
        queries: Path to the query list file with camera intrinsics.
        retrieval: Path to the retrieval results file.
        features: Path to the features directory for the reference images.
        matches: Path to the matches directory for the reference-query image pairs.
        results: Path to the output file for pose predictions.  
        ransac_thresh: RANSAC inlier threshold in pixels.
        covisibility_clustering: Whether to use covisibility clustering for pose estimation.
        prepend_camera_name: Whether to prepend the camera name to the query image name in the results.
        config: Optional configuration dictionary for the localizer.
        solver: Solver to use for pose estimation (from Solvers enum).
        features_query_path: Optional path to the features directory for the query images. If None, it defaults to the features path.
        gravity_vector_world: Optional gravity vector in world coordinates for upright solvers. Defaults to [0, 1, 0] if not provided.
        parallel: Number of parallel processes to use for query localization. If -1, uses all available cores.
    Returns:
        Writes the estimated poses to the results file in the format:
        <query_image_name> <qw> <qx> <qy> <qz> <tx> <ty> <tz> <focal_length> <time_taken>  
    """
    # Path assertions
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    # Data parsing
    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)
    assert isinstance(reference_sfm_path, str) or isinstance(reference_sfm_path, Path) or isinstance(reference_sfm_path, pycolmap.Reconstruction), \
        "reference_sfm_path should be a path to the reconstruction file"

    # Output structure set
    output_dict = {
        "cam_from_world": {},
        "time": {},
        "focal": {},
    }
    logs = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }

    # Read reference SfM model
    if not isinstance(reference_sfm_path, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm_path)
    else:
        reference_sfm = reference_sfm_path

    # Compute the normals for solvers that require them
    model_normals_dict = None
    if solver in [Solvers.P1P_AC, Solvers.UP1P_SIFT, Solvers.UP1PF_AC, Solvers.UP2PFORI]:
        t0 = time.time()
        model_normals_dict = estimate_normals(reference_sfm, neighbors=200)
        print("\tNormals estimated in %.2f" % (time.time()-t0))

    # Run the localization in parallel or sequentially
    if parallel > 0:
        ###################
        # Parallel processing
        ###################
        print(f"Running with parallelization: {parallel} processes")

        # Delete the sfm object to avoid pickling issues in multiprocessing
        del reference_sfm

        # Set up worker arguments for parallel processing
        config = {"estimation": {"ransac": {"max_error": ransac_thresh}}}
        worker_args = [
            (qname, qcam, retrieval_dict, features, matches, 
            solver, features_query_path, gravity_vector_world)
            for qname, qcam in queries
        ]

        # Run parallel pool with initializer to load SfM model and normals
        global worker_data
        worker_data = {}
        with Pool(processes=parallel, initializer=_init_worker, initargs=(reference_sfm_path, config, model_normals_dict)) as pool:
            outputs = pool.imap_unordered(_process_query_worker, worker_args)
            
            for result in tqdm(outputs, total=len(queries)):
                if result is None:
                    print("Failed")
                    continue
                    
                qname, ret, log = result
                output_dict["cam_from_world"][qname] = ret["cam_from_world"]
                output_dict["time"][qname] = ret["time"]
                output_dict["focal"][qname] = ret["camera"]

            pool.close()
            pool.join()
    else:
        print("Running without parallelization")

        # Create a mapping from database image names to their IDs for quick lookup and localizer
        db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
        config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
        localizer = QueryLocalizer(reference_sfm, config)

        # Run sequential localization for each query
        for qname, qcam in tqdm(queries):
            if qname not in retrieval_dict:
                logger.warning(f"No images retrieved for query image {qname}. Skipping...")
                continue
            db_names = retrieval_dict[qname]
            db_ids = [db_name_to_id[n] for n in db_names]

            ret, log = pose_from_gcransac(
                localizer, qname, qcam, db_ids, features, matches, solver=solver, features_query_path=features_query_path,
                model_normals_dict=model_normals_dict, gravity_vector_world=gravity_vector_world
            )
            output_dict["cam_from_world"][qname] = ret["cam_from_world"]
            output_dict["time"][qname] = ret["time"]
            output_dict["focal"][qname] = ret["camera"]
                
            log["covisibility_clustering"] = covisibility_clustering
            logs["loc"][qname] = log

    # Parse output and write results to file
    logger.info(f"Localized {len(output_dict['cam_from_world'])} / {len(queries)} images.")
    with open(results, "w") as fil:
        for query in output_dict["cam_from_world"].keys():
            t, f, tim = output_dict["cam_from_world"][query], output_dict["focal"][query], output_dict["time"][query]
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            fil.write(f"{name} {qvec} {tvec} {f} {tim}\n")
