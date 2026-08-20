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
#include <vector>
#include "../maths/affine2sift.h"


namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{			
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class UP2PfOriSolver : public SolverEngine
			{
			public:
				UP2PfOriSolver()
				{
				}

				~UP2PfOriSolver()
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
					return 2;
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
			
			OLGA_INLINE bool UP2PfOriSolver::estimateModel(
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
				
				// If no sample is provided use all points
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				std::vector<Eigen::Vector2d> kps, p_ref_origs;
				std::vector<Eigen::Vector3d> normals, t_refs;
				std::vector<double> angle_refs, angle_querys, ds, fs;
                std::vector<Eigen::Matrix3d> R_refs;
				Eigen::Matrix3d Rxz;

				for ( int i = 0; i < 2; i++ )
				{
					const size_t idx = (sample_ == nullptr ? i : sample_[i]);
					
					Eigen::Vector3d point_aux;
					Eigen::Matrix2d aff;
					Eigen::Matrix3d R_aux;
					double angle_ref, angle_query;
					
					kps.push_back(Eigen::Vector2d(data_.at<double>(idx, 0), data_.at<double>(idx, 1)));
					point_aux << data_.at<double>(idx, 2), data_.at<double>(idx, 3), data_.at<double>(idx, 4);

					p_ref_origs.push_back(point_aux.hnormalized());
					ds.push_back(point_aux(2));
					
					fs.push_back(data_.at<double>(idx, 5));
					normals.push_back(Eigen::Vector3d(data_.at<double>(idx, 6), data_.at<double>(idx, 7), data_.at<double>(idx, 8)));

					// Angle ref, scale ref, angle query, scale query
					angle_ref = data_.at<double>(idx, 9); angle_query = data_.at<double>(idx, 11);

					angle_refs.push_back(angle_ref);
					angle_querys.push_back(angle_query);
					
					// The next 3*3 elements are the rotation matrix of the reference frame in a row-major order
					Rxz << data_.at<double>(idx, 13), data_.at<double>(idx, 14), data_.at<double>(idx, 15),
						   data_.at<double>(idx, 16), data_.at<double>(idx, 17), data_.at<double>(idx, 18),
					       data_.at<double>(idx, 19), data_.at<double>(idx, 20), data_.at<double>(idx, 21);

					///////////////////////////////////////////////////////////////////////////////////////
					// Experimental, T_ref
					R_aux << data_.at<double>(idx, 22), data_.at<double>(idx, 23), data_.at<double>(idx, 24),
							 data_.at<double>(idx, 25), data_.at<double>(idx, 26), data_.at<double>(idx, 27),
							 data_.at<double>(idx, 28), data_.at<double>(idx, 29), data_.at<double>(idx, 30);

					R_refs.push_back(R_aux);
					t_refs.push_back(Eigen::Vector3d(data_.at<double>(idx, 31), data_.at<double>(idx, 32), data_.at<double>(idx, 33)));
				}
				
				///////////////////////////////////////////////////////////////////////////////////////
        		std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> output;
                output = ECCV2026::solver_up2pf_ori(
							R_refs,
							t_refs,
							fs, 
							angle_refs,
							angle_querys,
							p_ref_origs, 
							ds,
							normals, 
							kps,
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