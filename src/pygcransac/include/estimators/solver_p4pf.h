// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include "../maths/re3q3.h"
#include "../relative_pose/bundle.h"
#include "solver_engine.h"
#include "eccv2026/eccv2026.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class P4PfSolver : public SolverEngine
			{
			public:
				P4PfSolver()
				{
				}

				~P4PfSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 8;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			};


			OLGA_INLINE bool P4PfSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// Check if we have enough points for the bundle adjustment
				if (sample_number_ < sampleSize())
					return false;

				// If no sample is provided use all points
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				std::vector<Eigen::Vector2d> kps;
				std::vector<Eigen::Vector3d> points;
				const size_t columns = data_.cols;
				// std::cout << "COLUMNS: " << columns <<  "x" << data_.rows << std::endl;
				// std::cout << "\nDATA: ";

				for (int i = 0; i < 4; ++i) {
					const size_t idx = (sample_ == nullptr ? i : sample_[i]);
					Eigen::Vector2d kp;
					kp << data_.at<double>(idx, 0), data_.at<double>(idx, 1);
					kps.push_back(kp);
					Eigen::Vector3d point;
					point << data_.at<double>(idx, 2), data_.at<double>(idx, 3), data_.at<double>(idx, 4);
					points.push_back(point);
					// std::cout << std::endl << kp << std::endl << point << std::endl;
				}

				// Iterating through the possible poses and optimizing each
				// Apply bundle adjustment
				std::vector<Eigen::Matrix3d> output_R;
				std::vector<Eigen::Vector3d> output_T;
				// std::vector<pose_lib::CameraPose> output;
				std::vector<double> output_f;

				int n_sols = ECCV2026::p4pf_wrapper(
					kps, // Keypoints
					points, // 3d Points
					&output_R,
					&output_T,
					&output_f, 
					false); 

				for (size_t i = 0; i<n_sols; i++)
				{					
					Model model;
					model.descriptor.resize(4, 4);
					model.descriptor.block(0, 0, 3, 3) << output_R[i];
					model.descriptor.row(3).setZero();
					model.descriptor.col(3) << output_T[i], output_f[i];
					models_.emplace_back(model);
				}


				return models_.size();
			}

		}
	}
}