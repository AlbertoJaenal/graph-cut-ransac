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
#include "../maths/affine2sift.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class UP1SIFTSolver : public SolverEngine
			{
			public:
				UP1SIFTSolver()
				{
				}

				~UP1SIFTSolver()
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
					return 1;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return true;
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
			
			OLGA_INLINE bool UP1SIFTSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const double * data_ptr = reinterpret_cast<double *>(data_.row(sample_[0]).data);

				Eigen::Vector2d p_query;
				Eigen::Vector3d normal, X, t_ref;
				Eigen::Matrix2d aff;
                Eigen::Matrix3d Rxz, R_ref;
				double s_ref,c_ref, s_query,c_query, q, d, f;
				double angle_ref, angle_query;

				p_query << data_ptr[0], data_ptr[1];
				X << data_ptr[2], data_ptr[3], data_ptr[4];
				f  = data_ptr[5];

				normal << data_ptr[6], data_ptr[7], data_ptr[8];
				// Angle ref, scale ref, angle query, scale query
				angle_ref = data_ptr[9]; angle_query = data_ptr[11];
				q = data_ptr[12] / data_ptr[10]; // sqrt(A.determinant()): scale_query / scale_ref

				Rxz   << data_ptr[13], data_ptr[14], data_ptr[15],
					     data_ptr[16], data_ptr[17], data_ptr[18],
					     data_ptr[19], data_ptr[20], data_ptr[21];
				R_ref << data_ptr[22], data_ptr[23], data_ptr[24],
						 data_ptr[25], data_ptr[26], data_ptr[27],
						 data_ptr[28], data_ptr[29], data_ptr[30];
				t_ref << data_ptr[31], data_ptr[32], data_ptr[33];

				 std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>> out = 
				 			ECCV2026::solver_up1p_sift(R_ref, t_ref, p_query, X, Rxz, angle_ref, angle_query, q*f, normal);


				for ( int i = 0; i < out.first.size(); i++ )
				{
					Eigen::Matrix3d Rsoln = out.first[i];
					Eigen::Vector3d tsoln = out.second[i];
					
					Model model;
					model.descriptor.resize(3, 4);
					model.descriptor << Rsoln, tsoln;
					models_.emplace_back(model);
				}
				
				return models_.size();
			}

		}
	}
}