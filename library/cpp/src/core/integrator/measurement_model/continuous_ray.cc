#include "wavemap/core/integrator/measurement_model/continuous_ray.h"
#include <iostream>

namespace wavemap {
DECLARE_CONFIG_MEMBERS(ContinuousRayConfig,
                      (range_sigma)
                      (scaling_free)
                      (scaling_occupied)
                      (beam_selector_type));

bool ContinuousRayConfig::isValid(bool verbose) const {
  bool is_valid = true;

  is_valid &= IS_PARAM_GT(range_sigma, 0.f, verbose);

  is_valid &= IS_PARAM_GT(scaling_free, 0.f, verbose);
  is_valid &= IS_PARAM_GT(scaling_occupied, 0.f, verbose);

  return is_valid;
}

// FloatingPoint LieContinuousRay::computeWorstCaseApproximationError(
//     UpdateType update_type, FloatingPoint /*sphere_center_distance*/,
//     FloatingPoint cell_bounding_radius) const {
//   if (update_type == UpdateType::kFullyUnobserved ||
//       update_type == UpdateType::kFullyFree) {
//     return 0.f;
//   }

//   if (update_type == UpdateType::kFreeOrUnobserved) {
//     // TODO(victorr): Check this
//     return 0.5f * config_.scaling_free;
//   }

//   const FloatingPoint worst_dz = cell_bounding_radius;
//   const FloatingPoint worst_dz_error =
//     3.f / 8.f * config_.scaling_occupied / config_.range_sigma * worst_dz;

//   return worst_dz_error;
// }

bool LieContinuousRay::computeWorstCaseApproximationError(
    const SensorCoordinates& sensor_coordinates,
    const Point3D& r_pcenterz_c,
    const Point3D& r_pmaxz_c,
    const Point3D& r_pminz_c,
    const Eigen::Matrix4d& T_ac,
    UpdateType update_type,
    FloatingPoint bounding_sphere_radius) const {

  if (update_type == UpdateType::kFullyUnobserved ||
      update_type == UpdateType::kFullyFree) {
    return true;
  }

  // compute reprojected image index
  const auto image_index =
      projection_model_->imageToNearestIndex(sensor_coordinates.image);

  // if index is not within bounds, return 0 update
  if (!depth_image_->isIndexWithinBounds(image_index)) {
    return 0.f;
  }

  // compute T_cp_tilde using the depth image
  FloatingPoint d_p = depth_image_->at(image_index);

  // if depth image does not see anything within a reasonable nonzero range, or is non-finite, skip update
  // as we cannot construct a valid T_cp_tilde
  if (!std::isfinite(d_p) || d_p <= 1e-3f) {
    return false;
  }
  
  // compute r_pz_c_tilde, the reprojected voxel center on the depth image
  Point3D r_pz_c_tilde = projection_model_->sensorToCartesian(
    SensorCoordinates{sensor_coordinates.image, d_p});

  // now, compute T_ap and T_ap_tilde using gram-schmidt orthogonalization
  Eigen::Matrix4d T_cp = compute_gram_schmidt(r_pcenterz_c);
  Eigen::Matrix4d T_cp_max = compute_gram_schmidt(r_pmaxz_c);
  Eigen::Matrix4d T_cp_tilde = compute_gram_schmidt(r_pz_c_tilde);

  Eigen::Matrix4d T_ap = T_ac * T_cp;
  Eigen::Matrix4d T_apmax = T_ac * T_cp_max;
  Eigen::Matrix4d T_ap_tilde = T_ac * T_cp_tilde;
  Eigen::Matrix4d T_p_tilde_a = SE3::inverse(T_ap_tilde);

  // check that both transformations are valid
  if (!T_apmax.allFinite() || !T_ap_tilde.allFinite()) {
    LOG(WARNING) << "Invalid transformation matrix T_ap or T_ap_tilde, skipping update.";
    return 0.f;
  }

  // const Eigen::Matrix<double, 6, 1> delta_xi_pa =
  //   SE3::Log(
  //     T_p_tilde_a * (T_ap));
  // const Eigen::Matrix<double, 6, 1> delta_x_pa = 
  //   SE3::leftJacobianNonScrew(delta_xi_pa) * delta_xi_pa;
  const Eigen::Matrix<double, 6, 1> delta_xi_pmaxa =
    SE3::Log(
      SE3::inverse(T_apmax) * (T_ap));
  const Eigen::Matrix<double, 6, 1> delta_x_pmaxa = 
    SE3::leftJacobianNonScrew(delta_xi_pmaxa) * delta_xi_pmaxa;

  // ensure that delta_xi_ap is finite, non-zero
  if (!delta_xi_pmaxa.allFinite()) {
    LOG(WARNING) << "Invalid Lie algebra difference delta_xi_ap, skipping update.";
    return 0.f;
  }

  // get r_pz_c from T_cp_tilde and divide by the depth to get the unit direction
  Eigen::Vector3d u_pz_c = T_cp_tilde.block<3,1>(0,3);
  u_pz_c /= d_p;

  // const Eigen::Matrix4d T_pa_tilde = T_p_tilde_a;

  // retrieve the full covariance matrix
  const Eigen::Matrix<double, 6, 6> Eta_xi_ap =
      getFullCovarianceMatrix(image_index, T_cp_tilde, T_ac, u_pz_c);

  // use adjoint trick to transform covariance to T_pa frame
  const Eigen::Matrix<double, 6, 6> Ad_T_pa = 
      SE3::adjoint(T_p_tilde_a);
  const Eigen::Matrix<double, 6, 6> Eta_xi_pa = 
      Ad_T_pa * Eta_xi_ap * Ad_T_pa.transpose();

  const Eigen::Matrix<double, 6, 6> left_jac_inv_t_pa_tilde_ns = 
      SE3::leftJacobianInverseNonScrew(SE3::Log(T_p_tilde_a));

  // now, transform back to cartesian space using the left jacobian at T_pa_tilde
  const Eigen::Matrix<double, 6, 6> Eta_x_pa =
      left_jac_inv_t_pa_tilde_ns * Eta_xi_pa *
      left_jac_inv_t_pa_tilde_ns.transpose();

  // compute the inverse of the covariance matrix
  // const Eigen::Matrix<double, 6, 6> Eta_x_pa_inv = Eta_x_pa.inverse();

  // assuming the worst-case for all 6 dimensions, simply multiply delta_x_pmaxa by the diagonal of the covariance matrix and see if the sum is below the termination limit
  // const Eigen::Matrix<double, 1, 6> diag_Eta_x_pa_inv = Eta_x_pa_inv.diagonal().transpose();
  const Eigen::Matrix<double, 1, 6> diag_Eta_x_pa_inv = Eta_x_pa.diagonal().transpose();
  
  for (int i = 0; i < 6; i++) {
    if (i < 3) {
      if (((1 / diag_Eta_x_pa_inv(i)) * std::abs(delta_x_pmaxa(i)) * (3. / 16.)) > .1f) {
          return false;
        }
    }
    else {
      if (i < 5) {
        if (((1 / diag_Eta_x_pa_inv(i)) * bounding_sphere_radius * (3. / 16.)) > .1f) {
          return false;
        }
      }
      else {
        if (((1 / diag_Eta_x_pa_inv(i)) * bounding_sphere_radius * (3. / 8.)) > .1f) {
          return false;
        }
      }
    }
  }

  // if the maximal delta_x_pa is less than the norm of the delta_x_pmaxa, then we are outside the worst-case update bounds and should return false to indicate that we should refine further
  return true;
}

Eigen::Matrix4d LieContinuousRay::compute_gram_schmidt(const Point3D& r_pz_c) const {

  // helper function to compute gram-schmidt orthogonalization for an arbitrary point aligned to the general camera frame of Wavemap, such that
  // the 3rd basis vector p_c_3 is aligned with the ray from the camera origin to the point
  Vector3D p_c_3 = r_pz_c.normalized();

  // now, we align the 2nd basis vector p_c_2 with the 2-axis of the camera frame, which is "left" in wavemap's definition
  Vector3D u_2 = Vector3D::UnitY();
  Vector3D p_c_2 = (u_2 - (u_2.dot(p_c_3)) * p_c_3).normalized();

  // finally, compute the 1st basis vector as the cross product of the other two
  Vector3D p_c_1 = p_c_2.cross(p_c_3);

  // now, we can construct C_cp
  Eigen::Matrix3d C_cp = Eigen::Matrix3d::Identity();
  C_cp.col(0) = p_c_1.cast<double>();
  C_cp.col(1) = p_c_2.cast<double>();
  C_cp.col(2) = p_c_3.cast<double>();

  // finally, construct T_cp as a TransformationMatrix
  Eigen::Matrix4d T_cp = Eigen::Matrix4d::Identity();
  T_cp.template block<3,3>(0,0) = C_cp;
  T_cp.template block<3,1>(0,3) = r_pz_c.cast<double>();

  return T_cp;
}

FloatingPoint LieContinuousRay::computeUpdate(
    const SensorCoordinates& sensor_coordinates,
    const Point3D& r_pcenterz_c,
    const Point3D& r_pmaxz_c,
    const Eigen::Matrix4d& T_ac) const {
  switch (config_.beam_selector_type) {
    case BeamSelectorType::kNearestNeighbor: {
      const auto image_index =
          projection_model_->imageToNearestIndex(sensor_coordinates.image);
      return computeBeamUpdate(image_index, sensor_coordinates, T_ac, r_pcenterz_c, r_pmaxz_c);
    }
    case BeamSelectorType::kAllNeighbors: {
      FloatingPoint update = 0.f;
      const auto image_indices =
          projection_model_->imageToNearestIndices(sensor_coordinates.image);
      for (int neighbor_idx = 0; neighbor_idx < 4; ++neighbor_idx) {
        const Index2D& image_index = image_indices.col(neighbor_idx);
        update += computeBeamUpdate(image_index, sensor_coordinates, T_ac, r_pcenterz_c, r_pmaxz_c);
      }
      return update;
    }
    default:
      return 0.f;
  }
}

FloatingPoint LieContinuousRay::computeBeamUpdate(
    const Index2D& image_index,
    const SensorCoordinates& sensor_coordinates,
    const Eigen::Matrix4d& T_ac,
    const Point3D& r_pcenterz_c,
    const Point3D& r_pmaxz_c) const {

  // if index is not within bounds, return 0 update
  if (!depth_image_->isIndexWithinBounds(image_index)) {
    return 0.f;
  }

  // compute T_cp_tilde using the depth image
  FloatingPoint d_p = depth_image_->at(image_index);

  // if depth image does not see anything within a reasonable nonzero range, or is non-finite, skip update
  // as we cannot construct a valid T_cp_tilde
  if (!std::isfinite(d_p) || d_p <= 1e-3f) {
    return 0.f;
  }

  // LOG(INFO) << "Computing T_cp and T_cp_tilde using gram-schmidt orthogonalization.";
  
  // compute r_pz_c_tilde, the reprojected voxel center on the depth image
  Point3D r_pz_c_tilde = projection_model_->sensorToCartesian(
    SensorCoordinates{sensor_coordinates.image, d_p});

  // the voxel center itself
  // Point3D r_pz_c = r_pcenterz_c;

  // // the maximal corner
  // Point3D r_pz_c_max = r_pmaxz_c;

  // now, compute T_ap and T_ap_tilde using gram-schmidt orthogonalization
  Eigen::Matrix4d T_cp = compute_gram_schmidt(r_pcenterz_c);
  Eigen::Matrix4d T_cp_tilde = compute_gram_schmidt(r_pz_c_tilde);

  Eigen::Matrix4d T_ap = T_ac * T_cp;
  Eigen::Matrix4d T_ap_tilde = T_ac * T_cp_tilde;
  Eigen::Matrix4d T_p_tilde_a = SE3::inverse(T_ap_tilde);

  // check that both transformations are valid
  if (!T_ap.allFinite() || !T_ap_tilde.allFinite()) {
    LOG(WARNING) << "Invalid transformation matrix T_ap or T_ap_tilde, skipping update.";
    return 0.f;
  }

  const Eigen::Matrix<double, 6, 1> delta_xi_pa =
    SE3::Log(
      T_p_tilde_a * (T_ap));
  const Eigen::Matrix<double, 6, 1> delta_x_pa = 
    SE3::leftJacobianNonScrew(delta_xi_pa) * delta_xi_pa;

  // std::cout << print::eigen::oneLine(delta_x_pa) << " depth: " << d_p << " sensor depth: " << sensor_coordinates.depth << std::endl;
  // exit(0);

  // ensure that delta_xi_ap is finite, non-zero
  if (!delta_xi_pa.allFinite()) {
    LOG(WARNING) << "Invalid Lie algebra difference delta_xi_ap, skipping update.";
    return 0.f;
  }

  // get r_pz_c from T_cp_tilde and divide by the depth to get the unit direction
  Eigen::Vector3d u_pz_c = T_cp_tilde.block<3,1>(0,3);
  u_pz_c /= d_p;

  // const Eigen::Matrix4d T_pa_tilde = T_p_tilde_a;

  // retrieve the full covariance matrix
  const Eigen::Matrix<double, 6, 6> Eta_xi_ap =
      getFullCovarianceMatrix(image_index, T_cp_tilde, T_ac, u_pz_c);

  // use adjoint trick to transform covariance to T_pa frame
  const Eigen::Matrix<double, 6, 6> Ad_T_pa = 
      SE3::adjoint(T_p_tilde_a);
  const Eigen::Matrix<double, 6, 6> Eta_xi_pa = 
      Ad_T_pa * Eta_xi_ap * Ad_T_pa.transpose();

  const Eigen::Matrix<double, 6, 6> left_jac_inv_t_pa_tilde_ns = 
      SE3::leftJacobianInverseNonScrew(SE3::Log(T_p_tilde_a));

  // now, transform back to cartesian space using the left jacobian at T_pa_tilde
  const Eigen::Matrix<double, 6, 6> Eta_x_pa =
      left_jac_inv_t_pa_tilde_ns * Eta_xi_pa *
      left_jac_inv_t_pa_tilde_ns.transpose();

  // compute the inverse of the covariance matrix
  const Eigen::Matrix<double, 6, 6> Eta_x_pa_inv = Eta_x_pa.inverse();

  // as an approximation, compute the covariance matrix by simply taking the reciprocal of the diagonals
  // now, retrieve diagonal elements of the full covariance as a vector
  // const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap = Eta_xi_ap.diagonal();

  // // loop through the diagonal elements and assign them to the inverse matrix
  // Eigen::Matrix<double, 6, 6> Eta_x_pa_inv = Eigen::Matrix<double, 6, 6>::Zero();

  // for (int i = 0; i < 6; i++) {
  //   Eta_x_pa_inv(i, i) = 1 / diag_Eta_xi_ap(i);
  // }

  // calculate the mahalanobis distance
  FloatingPoint mahalanobis_distance = std::sqrt(delta_x_pa.transpose() * Eta_x_pa_inv * delta_x_pa);

  // first, check if we are behind the mean endpoint, if so, return 0 update

  // first, check if delta_x_pa is within the bounds of the endpoint uncertainty
  if (mahalanobis_distance < 4.48f) {  // 4.48 is the chi2inv(0.9973, 6) value

    //if so, compute the full beam update
    FloatingPoint sample_update = computeFullBeamUpdate(delta_x_pa, Eta_x_pa);

    return sample_update;
  }
  else {
    // first, check if we are within the best-case update bounds (half-voxel displacement)
    
    // compute the normalized delta_x_pa
    Eigen::Matrix<double, 6, 1> normalized_delta_x_pa = delta_x_pa / delta_x_pa.norm();

    // now, compute the maximal displacement to be within the multivariate 3-sigma bounds
    // as per maximal_delta_x = sqrt(chi2inv(0.9973, 6))**2 / (normalized_delta_x_pa.transpose() * Eta_x_pa_inv * normalized_delta_x_pa)
    float maximal_delta_x = 4.48f / std::sqrt(normalized_delta_x_pa.transpose() * Eta_x_pa_inv * normalized_delta_x_pa);

    // compute the maximal delta_x_pa from center to corner
    // Eigen::Matrix4d T_cpmax = compute_gram_schmidt(r_pz_c_max);
    Eigen::Matrix4d T_ap_max = T_ac * compute_gram_schmidt(r_pmaxz_c);

    Eigen::Matrix<double, 6, 1> delta_xi_pmaxa =
      SE3::Log(
        SE3::inverse(T_ap_max) * (T_ap));
    Eigen::Matrix<double, 6, 1> delta_x_pmaxa = 
      SE3::leftJacobianNonScrew(delta_xi_pmaxa) * delta_xi_pmaxa;

    // compare the maximal delta_x_pa to the norm of the delta_x_pmaxa
    if ((maximal_delta_x < delta_x_pmaxa.norm()) && (delta_x_pa.norm() < delta_x_pmaxa.norm())) {
      
      // we are within the best-case update bounds, so return a completely occupied update
      return std::log(.9999 / (1.f - .9999)); 
    }

    // now, check if we are within the bounds for every dimension aside from depth
    // FloatingPoint mahalanobis_distance_no_depth = std::sqrt(
    //   delta_x_pa.head<5>().transpose() *
    //   Eta_x_pa_inv.topLeftCorner<5,5>() *
    //   delta_x_pa.head<5>());

    if (true) {  // 3.65 is the chi2inv(0.9973, 5) value
      // we are within bounds for all but depth, return free space update if delta_x_pa(5) < 0
      if (delta_x_pa(5) < 0.f) {
        // return computeFreeSpaceBeamUpdate();
        return computeFullBeamUpdate(delta_x_pa, Eta_x_pa);
      }
    }

    // else, we are outside the bounds or behind the surface, return 0 update
    return 0.f;
  }
}

Eigen::Matrix<double, 6, 6> LieContinuousRay::getFullCovarianceMatrix(const Index2D& image_index, const Eigen::Matrix4d& T_cp_tilde, const Eigen::Matrix4d& T_ac, const Eigen::Vector3d& u_pz_c) const {
  // first, get the depth uncertainty at image_index
  const FloatingPoint sigma_d = depth_uncertainty_image_->at(image_index);

  // stack u_pz_c into a 6D linear algebra with zeros for rotation such that [0; u_pz_c]
  Eigen::Matrix<double, 6, 1> u_pz_c_6d = Eigen::Matrix<double, 6, 1>::Zero();
  u_pz_c_6d.block<3,1>(3,0) = u_pz_c;

  // now, define a dilation matrix for the depth uncertainty
  // Eigen::Matrix<double, 6, 1> L = Eigen::Matrix<double, 6, 1>::Zero();
  // L(5) = 1.f;

  // the inverse left jacobian to map it to the Lie algebra
  Eigen::Matrix<double, 6, 6> J_left_inv_non_screw =
      SE3::leftJacobianInverseNonScrew(
        SE3::Log(T_cp_tilde));

  // the right jacobian
  Eigen::Matrix<double, 6, 6> J_left =
      SE3::leftJacobian(
        SE3::Log(T_cp_tilde));

  // adjoint of T_ac
  Eigen::Matrix<double, 6, 6> Ad_T_ac =
      SE3::adjoint(T_ac);

  // then, compute the total uncertainty
  // const Eigen::Matrix<double, 6, 6> Eta_xi_ap = Eta_ac + (sigma_d * sigma_d) * J_left * J_left_inv_non_screw * u_pz_c_6d * u_pz_c_6d.transpose() * J_left_inv_non_screw.transpose() * J_left.transpose();
  const Eigen::Matrix<double, 6, 6> Eta_xi_ap =
      *pose_covariance_ + Ad_T_ac * J_left * J_left_inv_non_screw * u_pz_c_6d * u_pz_c_6d.transpose() * J_left_inv_non_screw.transpose() * J_left.transpose() * Ad_T_ac.transpose() * (sigma_d * sigma_d);

  // throw an error if covariance matrix is not positive definite
  // Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt_of_Eta_xi_ap(Eta_xi_ap);
  // DCHECK_EQ(llt_of_Eta_xi_ap.info(), Eigen::Success)
  //     << "Covariance matrix is not positive definite!";

  return Eta_xi_ap;
}

FloatingPoint LieContinuousRay::computeFullBeamUpdate(
    const Eigen::Matrix<double, 6, 1>& delta_xi_ap,
    const Eigen::Matrix<double, 6, 6>& Eta_xi_ap) const {

  // now, retrieve diagonal elements of the full covariance as a vector
  const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap =
      Eta_xi_ap.diagonal();

  double support_fac = 1.0;

  // based on a maximum depth of voxel_res * 12, as per supereight2
  const double max_depth_sigma = .025 * 12;

  const double depth_sigma = std::sqrt(diag_Eta_xi_ap(5));
  support_fac = std::min(1.0, max_depth_sigma / depth_sigma);

  // calculate the depth contribution using the last element of the diagonal
  const double f_depth =
      (delta_xi_ap(5)) / std::sqrt(diag_Eta_xi_ap(5));
  double scaled_contrib =
      ApproximateGaussianDistribution::cumulative(f_depth) -
      0.5f * ApproximateGaussianDistribution::cumulative(f_depth - 3.f) - 0.5f;
  // double scaled_contrib = 1.f;

  // now, iterate through the other elements to compute the other contributions using C(g + 3.f) - C(g - 3.f)
  for (int i = 0; i < 5; ++i) {
    const double f_other =
        (delta_xi_ap(i)) / std::sqrt(diag_Eta_xi_ap(i));
    
    double support_fac = 1.0;

    // if i is {0, 1, 2}, then define the maximum angular sigma as 5 degrees
    if (i < 3) {
      const double max_angular_sigma = .1f * M_PI / 180.f; // 5 degrees in radians
      const double angle_sigma = std::sqrt(diag_Eta_xi_ap(i));

      support_fac = std::min(1.0, max_angular_sigma / angle_sigma);
    }
    // we define the maximum x, y sigma as .05 meters
    else {
      const double max_xy_sigma = 0.01f;
      const double xy_sigma = std::sqrt(diag_Eta_xi_ap(i));
      support_fac = std::min(1.0, max_xy_sigma / xy_sigma);
    }
    scaled_contrib *=
        (ApproximateGaussianDistribution::cumulative(f_other + 3.f * support_fac) -
         ApproximateGaussianDistribution::cumulative(f_other - 3.f * support_fac));
  }

  const double p = scaled_contrib + 0.5f;
  // std::cout << " scaled_contrib: " << scaled_contrib << std::endl;
  const double log_odds = std::log(p / (1.f - p));
  DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
  return log_odds;
}

FloatingPoint LieContinuousRay::computeFreeSpaceBeamUpdate() const {
  // const FloatingPoint p = -0.5f * config_.scaling_free + 0.5f;
  const FloatingPoint p = 0.01f; // a small free space update
  const FloatingPoint log_odds = std::log(p / (1.f - p));
  DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
  return log_odds;
}

}  // namespace wavemap
