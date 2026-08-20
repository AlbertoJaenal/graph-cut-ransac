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

#include "solver_engine.h"
#include "eccv2026/eccv2026.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class UP2PESolver : public SolverEngine
			{
			public:
				UP2PESolver()
				{
				}

				~UP2PESolver()
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
					return 2;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
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

			protected:
				/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
				int solve_quadratic_real(double a, double b, double c, double roots[2]) const
				{
					double b2m4ac = b * b - 4 * a * c;
					if (b2m4ac < 0)
						return 0;

					double sq = std::sqrt(b2m4ac);

					// Choose sign to avoid cancellations
					roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
					roots[1] = c / (a * roots[0]);

					return 2;
				}
			};
			
			OLGA_INLINE bool UP2PESolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				const double * data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t columns = data_.cols;

				// Eigen::Vector2d x[2];
				// Eigen::Vector3d X[2];
				std::vector<Eigen::Vector2d> kps;
				std::vector<Eigen::Vector3d> points;
				Eigen::Matrix3d Rxz;
				
				for ( int k = 0; k < 2; k++ )
				{
					double * data_ptr;
					if (sample_ == nullptr)
						data_ptr = reinterpret_cast<double *>(data_.row(k).data);
					else
						data_ptr = reinterpret_cast<double *>(data_.row(sample_[k]).data);

					Eigen::Vector2d kp(data_ptr[0], data_ptr[1]);
					Eigen::Vector3d point(data_ptr[2], data_ptr[3], data_ptr[4]);
					Rxz << data_ptr[5],  data_ptr[6],  data_ptr[7],
						   data_ptr[8],  data_ptr[9],  data_ptr[10],
						   data_ptr[11], data_ptr[12], data_ptr[13];
					
					kps.push_back(kp);
					points.push_back(point);
				}
				
				std::vector<Eigen::Matrix3d> output_R;
				std::vector<Eigen::Vector3d> output_T;

				ECCV2026::up2p_wrapper(kps, points, Rxz, &output_R, &output_T);

				models_.clear();
				for (int i = 0; i < output_R.size(); ++i) {

					Model model;
					model.descriptor.resize(3, 4);
					model.descriptor << output_R[i], output_T[i];
					models_.emplace_back(model);
				}
				
				return models_.size();
			}
		}
	}
}