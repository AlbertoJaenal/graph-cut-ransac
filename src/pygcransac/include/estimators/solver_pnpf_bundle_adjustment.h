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
#include "perspective_n_point_f_estimator.h"
#include <iostream>
#include <opencv2/core/eigen.hpp>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{

		struct PoseResult {
			double f;
			Eigen::Matrix3d R;
			Eigen::Vector3d t;
		};

		Eigen::MatrixXd build_A_matrix(const std::vector<Eigen::Vector2d>& pts2d,
                                       const std::vector<Eigen::Vector3d>& pts3d,
                                       const Eigen::Matrix3d& T2d,
                                       const Eigen::Matrix4d& T3d,
                                       const Eigen::VectorXd& weights) {
            // Build the DLT matrix A for weighted least squares
            // Each correspondence contributes 2 rows
            const int n = static_cast<int>(pts2d.size());
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 12);
            
            for (int i = 0; i < n; ++i) {
                // Normalize coordinates
                Eigen::Vector3d p2n = T2d * pts2d[i].homogeneous();
                Eigen::Vector4d p3n = T3d * pts3d[i].homogeneous();
                
                double u = p2n(0), v = p2n(1);
                double X = p3n(0), Y = p3n(1), Z = p3n(2), W = p3n(3);
                
                // Apply weights to equations
                double w = weights(i);

                // u equation: [X Y Z W  0 0 0 0  -u*X -u*Y -u*Z -u*W] * p = 0
                A.row(2 * i)     << w*X, w*Y, w*Z, w*W, 0, 0, 0, 0, -w*u*X, -w*u*Y, -w*u*Z, -w*u*W;
                
                // v equation: [0 0 0 0  X Y Z W  -v*X -v*Y -v*Z -v*W] * p = 0
                A.row(2 * i + 1) << 0, 0, 0, 0, w*X, w*Y, w*Z, w*W, -w*v*X, -w*v*Y, -w*v*Z, -w*v*W;
            }
            return A;
        }


		PoseResult solveDLTAndExtractInitialGuess(const std::vector<Eigen::Vector2d>& pts2d, 
												const std::vector<Eigen::Vector3d>& pts3d) {
			const int n = pts2d.size();
			
			// --- 1. Proper Normalization ---
			Eigen::Vector2d mean2d(0, 0);
			Eigen::Vector3d mean3d(0, 0, 0);
			for (int i = 0; i < n; ++i) {
				mean2d += pts2d[i];
				mean3d += pts3d[i];
			}
			mean2d /= n;
			mean3d /= n;

			double s2d = 0, s3d = 0;
			for (int i = 0; i < n; ++i) {
				s2d += (pts2d[i] - mean2d).norm();
				s3d += (pts3d[i] - mean3d).norm();
			}
			s2d = std::sqrt(2.0) / (s2d / n);
			s3d = std::sqrt(3.0) / (s3d / n);

			Eigen::Matrix3d T2d;
			T2d << s2d, 0, -s2d * mean2d.x(), 
				   0, s2d, -s2d * mean2d.y(), 
				   0, 0, 1;

			Eigen::Matrix4d T3d;
			T3d << s3d, 0, 0, -s3d * mean3d.x(), 
				   0, s3d, 0, -s3d * mean3d.y(), 
				   0, 0, s3d, -s3d * mean3d.z(), 
				   0, 0, 0, 1;

			// --- 2. Initial Pass ---
			Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
			Eigen::MatrixXd A = build_A_matrix(pts2d, pts3d, T2d, T3d, weights);
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
			Eigen::VectorXd p = svd.matrixV().col(11);
			
			// --- 3. Second Pass (Reweighted) ---
			Eigen::Matrix<double, 3, 4> P_init;
			P_init << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11];
			
			for (int i = 0; i < n; ++i) {
				Eigen::Vector4d p3n = T3d * pts3d[i].homogeneous();
				double z_val = P_init.row(2).dot(p3n);
				weights(i) = (std::abs(z_val) > 1e-6) ? (1.0 / std::abs(z_val)) : 1.0;
			}

			A = build_A_matrix(pts2d, pts3d, T2d, T3d, weights);
			svd.compute(A, Eigen::ComputeFullV);
			p = svd.matrixV().col(11);

			// --- 4. Denormalization ---
			Eigen::Matrix<double, 3, 4> P_norm;
			P_norm << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11];

			// Denormalize
			Eigen::Matrix<double, 3, 4> P = T2d.inverse() * P_norm * T3d;
			
			// Recovery of the scale factor MUST be based on the third row 
			// because the third row of the rotation matrix (r3) must have a norm of 1.
			double sc = P.row(2).leftCols<3>().norm();
			
			if (sc < 1e-9) {
				// Bad DLT solution
				return {0, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
			}
			P /= sc; // Now the matrix is in the form [f*r1, f*t1; f*r2, f*t2; r3, t3]

			int front_count = 0;
			for(int i = 0; i < n; ++i) {
				double z = P.row(2).dot(pts3d[i].homogeneous());
				if (z > 0) front_count++;
			}

			// If they are behind, the whole P matrix was flipped by SVD
			if (front_count < n / 2) {
				P = -P;
			}
			
			Eigen::Matrix3d M = P.block<3, 3>(0, 0);
			double f_est = (M.row(0).norm() + M.row(1).norm()) / 2.0;
			
			// VALIDATE focal length (relative!)
			if (f_est < 0.01 || f_est > 1000.0) {
				// Fallback: use rough estimate
				f_est = 1;
			}

			// Force Rotation into SO(3)
			Eigen::Matrix3d K_inv = Eigen::Vector3d(1.0/f_est, 1.0/f_est, 1.0).asDiagonal();
			Eigen::Matrix3d R_approx = K_inv * M;

			Eigen::JacobiSVD<Eigen::Matrix3d> svd_r(R_approx, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d R = svd_r.matrixU() * svd_r.matrixV().transpose();

			if (R.determinant() < 0) {
				R = svd_r.matrixU() * Eigen::Vector3d(1, 1, -1).asDiagonal() * svd_r.matrixV().transpose();
			}

			Eigen::Vector3d t = P.col(3);

			return {f_est, R, t};
		}

		double estimateRoughFocal(const std::vector<Eigen::Vector2d>& pts2d, 
								const std::vector<Eigen::Vector3d>& pts3d) {
			std::vector<double> focal_guesses;
			for (size_t i = 0; i < pts2d.size(); ++i) {
				// pts3d[i](2) is the Z-depth in the camera coordinate frame
				double depth = abs(pts3d[i](2));
				
				// Ensure point is in front of camera and at a reasonable distance
				if (depth > 1e-3) {
					// Horizontal focal: u = f * (X / Z)  => f = (u * Z) / X
					// We can use the norms for a quick isotropic estimate
					double norm2d = pts2d[i].norm();
					double norm3d_xy = pts3d[i].head<2>().norm();
					
					if (norm3d_xy > 1e-6) {
						focal_guesses.push_back(norm2d * depth / norm3d_xy);
					}
				}
			}

			if (focal_guesses.empty()) return 1.0; // Fallback if no valid points
			
			std::sort(focal_guesses.begin(), focal_guesses.end());
			return focal_guesses[focal_guesses.size() / 2]; // Return median

		}

		double getReprojectionError(const cv::Mat& obj_pts, const cv::Mat& img_pts,
									const cv::Mat& K, const cv::Mat& R, const cv::Mat& t) {
			cv::Mat projected(obj_pts.rows, 2, CV_64F);
			cv::projectPoints(obj_pts, R, t, K, cv::Mat(), projected);
			double err = 0;
			for (size_t i = 0; i < obj_pts.rows; ++i) {// Access coordinates by row and column index
				double obs_x = img_pts.at<double>(i, 0);
				double obs_y = img_pts.at<double>(i, 1);
				
				double dx = projected.at<double>(i, 0) - obs_x;
				double dy = projected.at<double>(i, 1) - obs_y;
				err += std::sqrt(dx*dx + dy*dy);
			}
			return err / obj_pts.rows;
		}


		class PnPfBundleAdjustment : public SolverEngine
			{
			protected:
				// The options for the bundle adjustment
				pose_lib::BundleOptions bundle_options;

			public:
				PnPfBundleAdjustment(
					const pose_lib::BundleOptions::LossType &loss_type_ = pose_lib::BundleOptions::LossType::TRUNCATED,
					const size_t &maximum_iterations_ = 25)
				{
					bundle_options.loss_type = loss_type_;
					bundle_options.max_iterations = maximum_iterations_;
					// pose_lib::BundleOptions bundle_options;
					// bundle_options.max_iterations = 25;
					// bundle_options.loss_scale = 1;
					// bundle_options.loss_type = pose_lib::BundleOptions::LossType::CAUCHY;
				}

				~PnPfBundleAdjustment()
				{
				}

				pose_lib::BundleOptions& getMutableOptions()
				{
					return bundle_options;
				}

				const pose_lib::BundleOptions& getOptions() const
				{
					return bundle_options;
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
					return 6;
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
			
			OLGA_INLINE bool PnPfBundleAdjustment::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// 1. Minimum check: Non-minimal solvers need more than the minimal 4 points
				if (sample_number_ < sampleSize()) return false;
				if (sample_ == nullptr) sample_number_ = data_.rows;

				std::vector<Eigen::Vector3d> inlier_object_points_; inlier_object_points_.reserve(sample_number_);
				std::vector<Eigen::Vector2d> inlier_image_points_;  inlier_image_points_.reserve(sample_number_);
				cv::Mat inlier_image_points(sample_number_, 2, CV_64F), inlier_object_points(sample_number_, 3, CV_64F);
				
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx = 
						sample_ == nullptr ? i : sample_[i];
					inlier_image_points_.emplace_back( data_.at<double>(idx, 0), data_.at<double>(idx, 1));
					inlier_object_points_.emplace_back(data_.at<double>(idx, 2), data_.at<double>(idx, 3), data_.at<double>(idx, 4));

					inlier_image_points.at<double>(i, 0)  = data_.at<double>(idx, 0); 
					inlier_image_points.at<double>(i, 1)  = data_.at<double>(idx, 1);
					inlier_object_points.at<double>(i, 0) = data_.at<double>(idx, 2); 
					inlier_object_points.at<double>(i, 1) = data_.at<double>(idx, 3); 
					inlier_object_points.at<double>(i, 2) = data_.at<double>(idx, 4);
				}

				// 2. Data Preparation (Keep it lean)
				// If models_ already has a hypothesis (from P4Pf), use it as a seed.
				// Otherwise, we need an initial guess.
				pose_lib::CameraPose pose;
				double focal;
				Eigen::Matrix3d rotation;
				Eigen::Vector3d translation;

				if (!models_.empty()) { models_.clear(); } // Clear any existing models since we will refine and return only one.

				PoseResult sol = solveDLTAndExtractInitialGuess(inlier_image_points_, inlier_object_points_);
				rotation = sol.R;
				translation = sol.t;
				pose = pose_lib::CameraPose(rotation, translation);
				focal = sol.f;

				// std::cout << "Initial DLT error " << getReprojectionError(inlier_object_points, inlier_image_points, 
				// 							K, 
				// 							cv::Mat(3, 3, CV_64F, rotation.data()), 
				// 							cv::Mat(3, 1, CV_64F, translation.data())) << " focal: " << focal << std::endl;

				// focal = estimateRoughFocal(inlier_image_points_, inlier_object_points_); //sol.f;
				
				// cv::Mat cv_rotation(3, 3, CV_64F, rotation.data()), // The estimated rotation matrix converted to OpenCV format
				// 	cv_translation(3, 1, CV_64F, translation.data()); // The estimated translation converted to OpenCV format
				// cv::Mat cv_rodrigues;
				// cv::Rodrigues(cv_rotation.t(), cv_rodrigues); 

				// try {
				// 	cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
				// 	K.at<double>(0,0) = focal;
				// 	K.at<double>(1,1) = focal;
				// 	// Applying numerical optimization to the estimated pose parameters
				// 	bool success = cv::solvePnP(inlier_object_points, // The object points
				// 								inlier_image_points, // The image points
				// 								K, // The camera's intrinsic parameters 
				// 								cv::Mat(), // An empty vector since the radial distortion is not known
				// 								cv_rodrigues, // The initial rotation
				// 								cv_translation, // The initial translation
				// 								true, // Use the initial values
				// 								cv::SOLVEPNP_EPNP); // Apply numerical refinement

				// 	// Convert the rotation vector back to a rotation matrix
				// 	cv::Rodrigues(cv_rodrigues, cv_rotation);

				// 	// Transpose the rotation matrix back
				// 	cv_rotation = cv_rotation.t();

				// 	pose = pose_lib::CameraPose(rotation, translation);
				// 	// std::cout << "Initial CV error " << getReprojectionError(inlier_object_points, inlier_image_points, 
				// 	// 							K, 
				// 	// 							cv::Mat(3, 3, CV_64F, rotation.data()), 
				// 	// 							cv::Mat(3, 1, CV_64F, translation.data())) << " focal: " << focal << std::endl;

				// } catch (const cv::Exception& e) {
				// 	std::cerr << "OpenCV exception during initial PnP estimation: " << e.what() << std::endl;
				// 	return false; // Skip this sample/iteration
				// }
				

				// 3. Fast Refinement
				// PoseLib's refiner is an iterative L-M solver. It is the proper "non-minimal" 
				// way to handle n-points with unknown focal length.
				pose_lib::refine_pnpf(
					data_, // All point correspondences
					sample_, // The sample, i.e., indices of points to be used
					sample_number_, // The size of the sample
					&pose, // The optimized pose
					&focal,
					bundle_options, // The bundle adjustment options
					weights_); // The weights for the weighted LSQ fitting
				
				// 4. Update the model
				models_.clear();
				Model refined_model;
				refined_model.descriptor = Eigen::Matrix4d::Identity();
				refined_model.descriptor.block(0, 0, 3, 3) << pose.R();
				refined_model.descriptor.col(3) << pose.t, focal;
				models_.emplace_back(refined_model);

				// cv::Mat Kl = cv::Mat::eye(3, 3, CV_64F);
				// Kl.at<double>(0,0) = focal;
				// Kl.at<double>(1,1) = focal;
				// std::cout << "Refined error " << getReprojectionError(inlier_object_points, inlier_image_points, 
				// 							Kl, 
				// 							cv::Mat(3, 3, CV_64F, pose.R().data()), 
				// 							cv::Mat(3, 1, CV_64F, pose.t.data())) << " focal: " << focal << std::endl << std::endl;
				return models_.size();
			}
		}
	}
}

