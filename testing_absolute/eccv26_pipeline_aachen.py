"""ECCV26 Absolute Pose Estimation Pipeline for Aachen dataset.

Endpoint task: Query image localization in Aachen v1.1 dataset using various
absolute pose solvers. Estimates camera pose (rotation + translation) and focal length.

Pipeline steps:
1. Extract NetVLAD global descriptors for image retrieval
2. Extract local features (SIFT or SuperPoint) for SfM reference and queries
3. Match features between images
4. Build SfM model via triangulation
5. Localize query images using different solvers
6. Evaluate against ground truth
"""
import argparse
from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)

from utils import Solvers, DATA_DIR, main_gcloc, evaluate_ftim
from utils.extract_scale_ori import extract_scale_ori

def run(args):
    """Main pipeline: extract features, match, triangulate, and localize.
    
    Args:
        args: Command-line arguments with dataset, output, and solver parameters
    """
    # Setup paths for Aachen dataset
    dataset = args.dataset
    images = dataset / "images_upright/"
    sift_sfm = dataset / "3D-models/aachen_v_1_1"  # Reference SfM model

    outputs = args.outputs  # where everything will be saved
    sfm_pairs = (outputs / f"pairs-db-covis{args.num_covis}.txt")
    loc_pairs = (outputs / f"pairs-query-netvlad{args.num_loc}.txt")

    # Pick feature extraction configuration
    retrieval_conf = extract_features.confs["netvlad"]
    if args.feature == "sp+lg":
        reference_sfm = outputs / "sfm_sp+lg"
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superpoint+lightglue"]
        requires_sso_extract = True

    elif args.feature == "sift":
        reference_sfm = outputs / "sfm_sift+NN"
        feature_conf = extract_features.confs["sift"]
        feature_conf["model"]["max_keypoints"] = 4000
        matcher_conf = match_features.confs["NN-ratio"]
        requires_sso_extract = False

    else:
        raise ValueError(f"Unknown feature type: {args.feature}")

    # Extract features and scale/orientation information
    features = extract_features.main(feature_conf, images, outputs)
    extract_scale_ori(features, images, requires_sso_extract=requires_sso_extract)

    # Build reference SfM and localization pairs
    if sfm_pairs.exists():
        pass  # Skip if already computed
    else:
        pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    if reference_sfm.exists():
        pass  # Skip if already computed
    else:
        triangulation.main(
            reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
        )

    # Extract global descriptors for retrieval and get query pairs
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )
    
    gravity_vector_world = [0, 1, 0]  # Gravity direction for Aachen

    # Test each solver multiple times
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
        print(f'Working on {solver}')
        for times_i in range(args.times):
            # Generate output filename
            results = (outputs /
                      f"Aachen-v1.1_hloc_{args.feature}_netvlad{args.num_loc}_{solver}_{times_i}.txt")
            if results.exists():
                print(f"Result {results} already exists, skipping...")
                continue
            
            # Run localization
            main_gcloc(
                reference_sfm,
                dataset / "queries/*_time_queries_with_intrinsics.txt",
                loc_pairs,
                features,
                loc_matches,
                results,
                solver=solver,
                gravity_vector_world=gravity_vector_world,
                parallel=8
            )

            # Evaluate results
            evaluate_ftim(dataset, results, do_print=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECCV26 pipeline for Aachen absolute pose estimation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR + "/aachen",
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
        default="outputs/aachen",
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
        default=50,
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
    run(args)