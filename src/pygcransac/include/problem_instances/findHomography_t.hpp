#pragma once
#include "gcransac_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "neighborhood/flann_neighborhood_graph.h"
#include "neighborhood/grid_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/napsac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"
#include "samplers/importance_sampler.h"
#include "samplers/adaptive_reordering_sampler.h"
#include "samplers/single_point_sampler.h"

#include "preemption/preemption_sprt.h"

#include "inlier_selectors/empty_inlier_selector.h"
#include "inlier_selectors/space_partitioning_ransac.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;


// A method for estimating a homography matrix given 2D-2D correspondences
template<class HomographyEstimator>
int findHomography_t_(
	// The 2D-2D point correspondences.
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &homography, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
	// Number of elements per point in the data matrix
    int element_number,
	// The spatial coherence weight used in the local optimization
	double spatial_coherence_weight, 
	// The inlier-outlier threshold
	double threshold, 
	// The RANSAC confidence. Typical values are 0.95, 0.99.
	double conf,
	// Maximum iteration number. I do not suggest setting it to lower than 1000.
	int max_iters,
	// Minimum iteration number. I do not suggest setting it to lower than 50.
	int min_iters,
	// A flag to decide if SPRT should be used to speed up the model verification. 
	// It is not suggested if the inlier ratio is expected to be very low - it will fail in that case.
	// Otherwise, it leads to a significant speed-up. 
	bool use_sprt, 
	// Expected inlier ratio for SPRT. Default: 0.1
	double min_inlier_ratio_for_sprt,
	// The identifier of the used sampler. 
	// Options: 
	//	(0) Uniform sampler 
	// 	(1) PROSAC sampler. The correspondences should be ordered by quality (e.g., SNN ratio) prior to calling this function. 
	//	(2) Progressive NAPSAC sampler. The correspondences should be ordered by quality (e.g., SNN ratio) prior to calling this function. 
	//	(3) Importance sampler from NG-RANSAC. The point probabilities should be provided.
	//	(4) Adaptive re-ordering sampler from Deep MAGSAC++. The point probabilities should be provided. 
	int sampler_id,
	// The identifier of the used neighborhood structure. 
	// 	(0) FLANN-based neighborhood. 
	// 	(1) Grid-based neighborhood.
	int neighborhood_id,
	// The size of the neighborhood.
	// If (0) FLANN is used, the size if the Euclidean distance in the correspondence space
	// If (1) Grid is used, the size is the division number, e.g., 2 if we want to divide the image to 2 in along each axes (2*2 = 4 cells in total)
	double neighborhood_size,
	// A flag determining if space partitioning from 
	// Barath, Daniel, and Gabor Valasek. "Space-Partitioning RANSAC." arXiv preprint arXiv:2111.12385 (2021).
	// should be used to speed up the model verification.
	bool use_space_partitioning,
	// The variance parameter of the AR-Sampler. It is only used if that particular sampler is selected.
	double sampler_variance,
	// The number of RANSAC iterations done in the local optimization
	int lo_number)
{
	int num_tents = correspondences.size() / element_number;
	cv::Mat points(num_tents, element_number, CV_64F, &correspondences[0]);
	
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat empty_point_matrix(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&empty_point_matrix, // The input points
			{ 	0, // The cell size along axis X in the source image
				0, // The cell size along axis Y in the source image
				0, // The cell size along axis X in the destination image
				0 }, // The cell size along axis Y in the destination image
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		if (use_space_partitioning && neighborhood_id != 0)
		{
			fprintf(stderr, "Space Partitioning only works with Grid neighorbood yet. Thus, setting neighborhood_id = 0.\n");
			neighborhood_id = 0;
		}

		// Using only the point coordinates and not the affine elements when constructing the neighborhood.
		cv::Mat point_coordinates = points(cv::Rect(0, 0, 4, points.rows));

		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&point_coordinates,
				{ (w1 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
					(h1 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
					(w2 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
					(h2 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&point_coordinates, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}

		// Checking if the neighborhood graph is initialized successfully.
		if (!neighborhood_graph->isInitialized())
		{
			AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
			delete neighborhood_graph_ptr;

			fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
			return 0;
		}
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	//const double max_image_diagonal =
	//	sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	HomographyEstimator estimator;
	Homography model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w1), // The width of the source image
				static_cast<double>(h1), // The height of the source image
				static_cast<double>(w2), // The width of the destination image
				static_cast<double>(h2) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ImportanceSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            sampler_variance));
	}
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling), 3 (NG-RANSAC sampling), 4 (AR-Sampler)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	utils::RANSACStatistics statistics;

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<HomographyEstimator> preemptive_verification(
			points,
			estimator);

		if (use_space_partitioning)
		{			
			inlier_selector::SpacePartitioningRANSAC<HomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<HomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<HomographyEstimator>,
				preemption::SPRTPreemptiveVerfication<HomographyEstimator>,
				inlier_selector::SpacePartitioningRANSAC<HomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		} else
		{
			inlier_selector::EmptyInlierSelector<HomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<HomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<HomographyEstimator>,
				preemption::SPRTPreemptiveVerfication<HomographyEstimator>,
				inlier_selector::EmptyInlierSelector<HomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		}
	}
	else
	{
		// Initializing an empty preemption
		preemption::EmptyPreemptiveVerfication<HomographyEstimator> preemptive_verification;

		if (use_space_partitioning)
		{			
			inlier_selector::SpacePartitioningRANSAC<HomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<HomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<HomographyEstimator>,
				preemption::EmptyPreemptiveVerfication<HomographyEstimator>,
				inlier_selector::SpacePartitioningRANSAC<HomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		} else
		{
			inlier_selector::EmptyInlierSelector<HomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<HomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<HomographyEstimator>,
				preemption::EmptyPreemptiveVerfication<HomographyEstimator>,
				inlier_selector::EmptyInlierSelector<HomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

			// Start GC-RANSAC
			gcransac.run(points,
				estimator,
				main_sampler.get(),
				&local_optimization_sampler,
				neighborhood_graph.get(),
				model,
				preemptive_verification,
				inlier_selector);

			statistics = gcransac.getRansacStatistics();
		}
	}

	homography.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			homography[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// Return the number of inliers found
	return num_inliers;
}
