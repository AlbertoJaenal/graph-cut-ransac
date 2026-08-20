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

#include "utils.h"
#include "solver_engine.h"
#include "eccv2026/eccv2026.h"


namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class UP1PfACSolver : public SolverEngine
			{
			protected:
				// mutable Eigen::Vector3d t0_eigen = Eigen::Vector3d::Zero();
				// mutable double s0 = 1.0;
				// mutable bool is_set = false;
				// mutable double f0 = 1.0;

			public:
				UP1PfACSolver()
				{
				}

				~UP1PfACSolver()
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
					return 1;
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
			
			OLGA_INLINE bool UP1PfACSolver::estimateModel(
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
				double f_ref;

				p_query << data_ptr[0], data_ptr[1];	
				X << data_ptr[2], data_ptr[3], data_ptr[4];
				f_ref = data_ptr[5];

				normal << data_ptr[6], data_ptr[7], data_ptr[8];
				aff << data_ptr[9], data_ptr[10], data_ptr[11], data_ptr[12];

				// The next 3*3 elements are the rotation matrix of the reference frame in a row-major order
				Rxz   << data_ptr[13], data_ptr[14], data_ptr[15],
					     data_ptr[16], data_ptr[17], data_ptr[18],
					     data_ptr[19], data_ptr[20], data_ptr[21];
				R_ref << data_ptr[22], data_ptr[23], data_ptr[24],
						 data_ptr[25], data_ptr[26], data_ptr[27],
						 data_ptr[28], data_ptr[29], data_ptr[30];
				t_ref << data_ptr[31], data_ptr[32], data_ptr[33];
				///////////////////////////////////////////////////////////////////////////////////////

        		std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> output;
                output = ECCV2026::solver_up1pf_ac(
							R_ref,
							t_ref,
							f_ref, 
							aff, 
							X.hnormalized(), 
							X(2),
							normal, 
							p_query,
							Rxz);
				Eigen::Matrix3d R_est = std::get<0>(output);
				Eigen::Vector3d t_est = std::get<1>(output);
				double f_est = std::get<2>(output);
				
				Model model;
				model.descriptor.resize(4, 4);
				model.descriptor.block(0, 0, 3, 3) << R_est;
				model.descriptor.row(3).setZero();
				model.descriptor.col(3) << t_est, f_est;
				models_.push_back(model);

				return models_.size();
			}
		}
	}
}