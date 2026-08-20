#pragma once

#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "utils.h"
#include "solver_engine.h"
#include "eccv2026/eccv2026.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for 
			class P35PfSolver : public SolverEngine
			{
			public:
				P35PfSolver()
				{
				}

				~P35PfSolver()
				{
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation.
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				static constexpr const char *name()
				{
					return "P3.5PF";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}
				
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool P35PfSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// Check if the sample size is correct, i.e.,
				// this solver can't solve the over-determined case. 
				constexpr size_t minimalSampleSize = sampleSize();
				if (sample_number_ != minimalSampleSize)
				{
					fprintf(stderr, "Method '%s' is used with incorrect sample size (%d instead of %d).\n",
						"P35PfSolver", sample_number_, minimalSampleSize);
					return false;
				}
				
				// The pointer to the data
				Eigen::Matrix<double, 2, 4> kps;
				Eigen::Matrix<double, 3, 4> points;
				const size_t columns = data_.cols;
				// std::cout << "COLUMNS: " << columns <<  "x" << data_.rows << std::endl;

				for (int i = 0; i < 4; ++i) {
					const size_t idx = (sample_ == nullptr ? i : sample_[i]);
					kps.col(i)    << data_.at<double>(idx, 0), data_.at<double>(idx, 1);
					points.col(i) << data_.at<double>(idx, 2), data_.at<double>(idx, 3), data_.at<double>(idx, 4);
					// std::cout << i << "-" << idx << "  ";
				}
				
							
				std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> output = ECCV2026::solver_p35pf(kps, points);

				
							
				if (std::get<2>(output) > 5e5) {
					return false;
				}
				
				Model model;
				model.descriptor.resize(4, 4);
				model.descriptor.block(0, 0, 3, 3) << std::get<0>(output);
				model.descriptor.row(3).setZero();
				model.descriptor.col(3) << std::get<1>(output), std::get<2>(output);
				
				models_.push_back(model);
				
				
				return true;
			}
		}
	}
}