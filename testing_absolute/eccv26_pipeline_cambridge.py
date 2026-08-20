"""ECCV26 Absolute Pose Estimation Pipeline for Cambridge Landmarks dataset.

Tests multiple absolute pose estimation solvers on the Cambridge Landmarks
dataset with 5 scenes. Estimates camera pose and focal length for each query image.

Dataset: CambridgeLandmarks_Colmap_Retriangulated (5 scenes: KingsCollege, etc.)
"""
import argparse
import os
from pathlib import Path

from hloc import (
    extract_features,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)

from utils import Solvers, DATA_DIR, main_gcloc
from utils.extract_scale_ori import extract_scale_ori
from utils.utils_cambridge import create_query_list_with_intrinsics, evaluate_cambridge, scale_sfm_images


SCENES = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch", "GreatCourt"]


def run_cambridge_scene(images, gt_dir, outputs, results, num_covis, num_loc, feature_name, solver):
    """Run absolute pose estimation pipeline for a single scene.
    
    Args:
        images: Path to scene images
        gt_dir: Path to ground truth model directory
        outputs: Output directory for features/matches
        results: Output file for pose predictions
        num_covis: Number of covisible images for SfM
        num_loc: Number of retrieved images for localization
        feature_name: Feature type ("sift" or "sp+lg")
        solver: Solver to use for pose estimation
    
    Returns:
        Path to reference SfM model
    """
    ref_sfm_sift = gt_dir / "model_train"
    test_list = gt_dir / "list_query.txt"

    outputs.mkdir(exist_ok=True, parents=True)
    ref_sfm_scaled = outputs / "sfm_sift_scaled"
    sfm_pairs = outputs / f"pairs-db-covis{num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{num_loc}.txt"
    query_list = outputs / "query_list_with_intrinsics.txt"
    retrieval_conf = extract_features.confs["netvlad"]
    
    # Select feature extraction configuration
    if feature_name == "sp+lg":
        ref_sfm = outputs / "sfm_sp+lg"
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superpoint+lightglue"]
        requires_sso_extract = True
    elif feature_name == "sift":
        ref_sfm = outputs / "sfm_sift+NN"
        feature_conf = extract_features.confs["sift"]
        feature_conf["model"]["max_keypoints"] = 4000
        matcher_conf = match_features.confs["NN-ratio"]
        requires_sso_extract = False
    else:
        raise ValueError(f"Unknown feature type: {feature_name}")
    
    # Create query list with intrinsics if not exists
    if not os.path.exists(query_list):
        create_query_list_with_intrinsics(
            gt_dir / "empty_all", query_list, test_list, ext=".txt", image_dir=images
        )
    
    # Get query sequences from test list
    with open(test_list, "r") as f:
        query_seqs = {q.split("/")[0] for q in f.read().rstrip().split("\n")}

    # Retrieve query image pairs
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        db_model=ref_sfm_sift,
        query_prefix=query_seqs,
    )

    # Extract local features for reference and query
    features = extract_features.main(feature_conf, images, outputs, as_half=True)
    extract_scale_ori(features, images, requires_sso_extract=requires_sso_extract)

    # Build reference SfM matches
    if sfm_pairs.exists():
        pass  # Skip if already computed
    else:
        pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    # Create SfM model or skip if exists
    if ref_sfm.exists():
        pass  # Skip if already computed
    else:
        scale_sfm_images(ref_sfm_sift, ref_sfm_scaled, images)
        triangulation.main(
            ref_sfm, ref_sfm_scaled, images, sfm_pairs, features, sfm_matches
        )

    # Localization matches
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )
    
    # Scene-specific gravity vectors (depends on camera orientation conventions)
    if "OldHospital" in str(gt_dir):
        gravity_vector_world = [0, 1, 0]
    elif "StMarysChurch" in str(gt_dir):
        gravity_vector_world = [0, -1, 0]
    elif ("GreatCourt" in str(gt_dir) or "KingsCollege" in str(gt_dir) or
          "ShopFacade" in str(gt_dir)):
        gravity_vector_world = [0, 0, 1]
    else:
        raise ValueError(f"Unknown gravity direction for scene {gt_dir}")

    # Run localization with specified solver
    main_gcloc(
        ref_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        prepend_camera_name=True,
        solver=solver,
        gravity_vector_world=gravity_vector_world,
        parallel=10
    )
    return ref_sfm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECCV26 pipeline for Cambridge Landmarks absolute pose estimation")
    parser.add_argument("--scenes", default=SCENES, choices=SCENES, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR + "/Cambridge",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        help="Number of times to run the experiment, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/cambridge",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for localization, default: %(default)s",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="sp+lg",
        choices=["sift", "sp+lg"],
        help="Feature type to use, default: %(default)s",
    )
    args = parser.parse_args()

    gt_dirs = args.dataset / "CambridgeLandmarks_Colmap_Retriangulated_1024px"
    logger.setLevel(30)
    
    # Run for each scene and solver
    for scene in args.scenes:
        for solver in [
            Solvers.UP2P_ERI,
            Solvers.P1P_AC,
            Solvers.UP1P_SIFT,
            Solvers.P4PF,
            Solvers.UP3PF,
            Solvers.P35PF,
            Solvers.UP1PF_AC,
            Solvers.UP2PFORI,
        ]:
            print(f'Working on scene "{scene}" with {solver}')
            for times_i in range(args.times):
                results = (args.outputs / scene / "output" /
                          f"out_poses_{args.feature}_{solver}_{times_i}.txt")
                final_result = (args.outputs / scene / "output" /
                               f"results_{args.feature}_{solver}_{times_i}.txt")
                
                if results.exists():
                    print(f"Result {results} already exists, skipping...")
                    continue
                
                # Run scene pipeline
                sfm_dir = run_cambridge_scene(
                    args.dataset / scene,
                    gt_dirs / scene,
                    args.outputs / scene,
                    results,
                    args.num_covis,
                    args.num_loc,
                    args.feature,
                    solver
                )

                # Evaluate results
                logger.info(f'Evaluating scene "{scene}".')
                os.makedirs(args.outputs / scene / "output", exist_ok=True)
                evaluate_cambridge(
                    gt_dirs / scene / "empty_all",
                    results,
                    args.dataset / scene,
                    gt_dirs / scene / "list_query.txt",
                    ext=".txt",
                    out_path=final_result
                )
                print()