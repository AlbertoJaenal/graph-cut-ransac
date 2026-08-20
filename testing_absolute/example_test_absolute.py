"""Example script for testing absolute pose estimation on a standard benchmark.

Demonstrates the full pipeline:
1. Setup paths for TUM or Reichstag datasets
2. Create or retrieve SfM model
3. Extract scale/orientation information
4. Run pose estimation with multiple solvers
5. Evaluate results against ground truth
"""
from pathlib import Path

from utils import Solvers, RANSAC_DIR, create_model_or_retrieve, main_gcloc, evaluate_generic
from utils.extract_scale_ori import extract_scale_ori


if __name__ == "__main__":
    # Configuration
    feat = "sp+lg"  # Feature type: "sp+lg" or "sift"
    
    # Construct dataset paths for Reichstag dataset.
    BASE = Path(RANSAC_DIR / "testing_absolute")

    map_data_path = BASE / "datasets/reichstag/mapping/"
    outputs = BASE / "outputs/reichstag/"
    outputs_query = BASE / "outputs/reichstag/query"

    images = map_data_path / "images"
    input_sfm_dir = map_data_path / "input_model"

    base_query = BASE / "datasets/reichstag/localization/"
    images_query = base_query / "images"

    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / feat
    sfm_pairs_query = outputs_query / "pairs-netvlad.txt"

    reference_sfm = outputs / feat
    
    # Create or retrieve SfM model and features
    model, feature_path, feature_query_path, match_query_path = \
        create_model_or_retrieve(images, images_query, outputs, outputs_query,
                                 sfm_pairs, sfm_pairs_query, sfm_dir, input_sfm_dir,
                                 feat=feat)
    
    print(f"Model: {model}")
    print(f"Features: {feature_path}")
    
    # Extract feature-level scale and orientation information
    extract_scale_ori(feature_path, images, requires_sso_extract=True)
    extract_scale_ori(feature_query_path, images_query, requires_sso_extract=True)

    print()

    # Test multiple solvers
    for solver in [
        Solvers.P3P,
        Solvers.UP2P_ERI,
        Solvers.P1P_AC,
        Solvers.UP1P_SIFT,
        Solvers.P4PF,
        Solvers.UP3PF,
        Solvers.P35PF,
        Solvers.UP1PF_AC,
        Solvers.UP2PFORI,
    ]:
        print(f"Testing solver: {solver}")
        
        # Setup gravity vector
        gravity_vector_world = [0, -1, 0]
        
        # Run localization (pose estimation)
        main_gcloc(reference_sfm,
                   Path(str(images_query).replace("images",
                                                  "queries_with_intrinsics") + ".txt"),
                   sfm_pairs_query,
                   feature_path,
                   match_query_path,
                   "results.txt",
                   solver=solver,
                   features_query_path=feature_query_path,
                   gravity_vector_world=gravity_vector_world,
                   parallel=8)
        
        # Evaluate results
        evaluate_generic(
            images_query,
            "results.txt")
        print()
