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
    return false;
  }

  // compute T_cp_tilde using the depth image
  FloatingPoint d_p = depth_image_->at(image_index);

  // if depth image does not see anything within a reasonable nonzero range, or is non-finite, skip update
  // as we cannot construct a valid T_cp_tilde
  if (!std::isfinite(d_p) || d_p <= 1e-3f) {
    return false;
  }

  // in a similar manner to wavemap, first compute the maximal delta_x_pa in each direction from the center, min, and max of the voxel

  // compute r_pz_c_tilde, the reprojected voxel center on the depth image
  Point3D r_pz_c_tilde = projection_model_->sensorToCartesian(
    SensorCoordinates{sensor_coordinates.image, d_p});
  
  // compute the point defined by the maximal spread from the voxel center
  Point3D r_pz_c_bounding = r_pcenterz_c + Vector3D::Constant(bounding_sphere_radius / std::sqrt(3.f));

  // now, compute T_ap and T_ap_tilde using gram-schmidt orthogonalization
  Eigen::Matrix4d T_cp = compute_gram_schmidt(r_pcenterz_c);
  Eigen::Matrix4d T_cp_max = compute_gram_schmidt(r_pmaxz_c);
  Eigen::Matrix4d T_cp_min = compute_gram_schmidt(r_pminz_c);
  Eigen::Matrix4d T_cp_bounding = compute_gram_schmidt(r_pz_c_bounding);
  Eigen::Matrix4d T_cp_tilde = compute_gram_schmidt(r_pz_c_tilde);

  Eigen::Matrix4d T_ap = T_ac * T_cp;
  Eigen::Matrix4d T_apmax = T_ac * T_cp_max;
  Eigen::Matrix4d T_apmin = T_ac * T_cp_min;
  Eigen::Matrix4d T_apbounding = T_ac * T_cp_bounding;
  Eigen::Matrix4d T_ap_tilde = T_ac * T_cp_tilde;
  Eigen::Matrix4d T_p_tilde_a = SE3::inverse(T_ap_tilde);

  // check that both transformations are valid
  if (!T_apmax.allFinite() || !T_ap_tilde.allFinite()) {
    LOG(WARNING) << "Invalid transformation matrix T_ap or T_ap_tilde, skipping update.";
    return false;
  }

  const Eigen::Matrix<double, 6, 1> delta_xi_pa =
    SE3::Log(
      SE3::inverse(T_ap) * (T_ap_tilde));
  const Eigen::Matrix<double, 6, 1> delta_x_pa = 
    SE3::leftJacobianNonScrew(delta_xi_pa) * delta_xi_pa;

  const Eigen::Matrix<double, 6, 1> delta_xi_pmaxa =
    SE3::Log(
      SE3::inverse(T_apmax) * (T_ap_tilde));
  const Eigen::Matrix<double, 6, 1> delta_x_pmaxa = 
    SE3::leftJacobianNonScrew(delta_xi_pmaxa) * delta_xi_pmaxa;

  const Eigen::Matrix<double, 6, 1> delta_xi_pbounding =
    SE3::Log(
      SE3::inverse(T_apbounding) * (T_ap_tilde));
  const Eigen::Matrix<double, 6, 1> delta_x_pbounding = 
    SE3::leftJacobianNonScrew(delta_xi_pbounding) * delta_xi_pbounding;

  const Eigen::Matrix<double, 6, 1> delta_xi_pmina =
    SE3::Log(
      SE3::inverse(T_apmin) * (T_ap_tilde));
  const Eigen::Matrix<double, 6, 1> delta_x_pmina = 
    SE3::leftJacobianNonScrew(delta_xi_pmina) * delta_xi_pmina;

  // ensure that delta_xi_ap is finite, non-zero
  if (!delta_xi_pmaxa.allFinite()) {
    LOG(WARNING) << "Invalid Lie algebra difference delta_xi_ap, skipping update.";
    return false;
  }

  // get r_pz_c from T_cp_tilde and divide by the depth to get the unit direction
  Eigen::Vector3d u_pz_c = T_cp_tilde.block<3,1>(0,3);
  u_pz_c /= d_p;

  // retrieve the full covariance matrix
  Eigen::Matrix<double, 6, 6> Eta_x_pa;
  Eigen::Matrix<double, 6, 6> Eta_x_pa_inv;

  // generate query pair
  std::pair<int, int> query_pair(image_index.x(), image_index.y());

  // check if the covariance matrix for this pixel has already been computed and cached
  if (covariance_map_.find(query_pair) != covariance_map_.end() && information_map_.find(query_pair) != information_map_.end()) {

    // if so, retrieve it from the map
    Eta_x_pa = covariance_map_.at(query_pair);
    Eta_x_pa_inv = information_map_.at(query_pair);
  }
  else {
    const Eigen::Matrix<double, 6, 6> Eta_xi_ap =
        getFullCovarianceMatrix(image_index, T_cp_tilde, T_ac, u_pz_c);

    // use adjoint trick to transform covariance to T_pa frame
    const Eigen::Matrix<double, 6, 6> Ad_T_pa = 
        SE3::adjoint(T_p_tilde_a);
    const Eigen::Matrix<double, 6, 6> Eta_xi_pa = 
        Ad_T_pa * Eta_xi_ap * Ad_T_pa.transpose();

    const Eigen::Matrix<double, 6, 6> left_jac_inv_t_pa_tilde_ns = 
        SE3::leftJacobianNonScrew(SE3::Log(T_p_tilde_a));

    // now, transform back to cartesian space using the left jacobian at T_pa_tilde
    Eta_x_pa = left_jac_inv_t_pa_tilde_ns * Eta_xi_pa * left_jac_inv_t_pa_tilde_ns.transpose();

    // compute the inverse of the covariance matrix
    Eta_x_pa_inv = Eta_x_pa.inverse();

    // cache the covariance matrix and information matrix for this pixel. make copies of the matrices to avoid issues with Eigen's lazy evaluation and potential dangling references
    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);

    if (lock.owns_lock()) {
      covariance_map_.emplace(query_pair, Eta_x_pa);
      information_map_.emplace(query_pair, Eta_x_pa_inv);
    }
  }

  // generate delta_x_pa_worst from the maximum compute between delta_x_pa, delta_x_pmaxa, delta_x_pmina

  Eigen::Matrix<double, 6, 1> delta_x_pa_worst;
  for (int i = 0; i < 6; ++i) {
    delta_x_pa_worst(i) = std::max(std::abs(delta_x_pa(i)), std::max(std::abs(delta_x_pmaxa(i)), std::abs(delta_x_pmina(i))));
    delta_x_pa_worst(i) = std::max(delta_x_pa_worst(i), std::abs(delta_x_pbounding(i)));
  }

  // now, compute the worst-case update using the information matrix and the worst-case delta_x_pa 

  // const FloatingPoint worst_angle =
  //     cell_bounding_radius / cell_to_sensor_distance;
  // const FloatingPoint worst_angular_error =
  //     3.f / 16.f * config_.scaling_free / config_.angle_sigma * worst_angle;

  // if (update_type == UpdateType::kFreeOrUnobserved ||
  //     update_type == UpdateType::kFullyFree) {
  //   return worst_angular_error;
  // }

  // const FloatingPoint worst_dz = cell_bounding_radius;
  // const FloatingPoint worst_dz_error =
  //     3.f / 8.f * config_.scaling_occupied / config_.range_sigma * worst_dz;

  // return std::max(worst_angular_error, worst_dz_error);

  double worst_error = 0;

  // iterate through each dimension and assign the worst-case error based on the covariance matrix diagonal and the worst-case delta_x_pa
  const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap = Eta_x_pa.diagonal();

  for (int i = 0; i < 6; ++i) {
    if (i < 5) {
      // for the first 3 dimensions, the worst-case approximation is 3 * sigma_i / 16 * delta_x_i, where sigma_i is the standard deviation of the i-th dimension
      worst_error = std::max(worst_error, (3.f / 16.f) * std::sqrt(diag_Eta_xi_ap(i)) * delta_x_pa_worst(i));
    }
    else{
      // for the last dimension, the worst-case approximation is 3 * sigma_i / 8 * delta_x_i, where sigma_i is the standard deviation of the i-th dimension
      worst_error = std::max(worst_error, (3.f / 8.f) * std::sqrt(diag_Eta_xi_ap(i)) * delta_x_pa_worst(i));
    }
  }

  // LOG(INFO) << "Computed worst-case approximation error: " << worst_error << " for delta_x_pa_worst: " << delta_x_pa_worst.transpose() << " and diag_Eta_xi_ap: " << diag_Eta_xi_ap;

  return worst_error < 1e-3f;
}

// bool LieContinuousRay::computeWorstCaseApproximationError(
//     const SensorCoordinates& sensor_coordinates,
//     const Point3D& r_pcenterz_c,
//     const Point3D& r_pmaxz_c,
//     const Point3D& r_pminz_c,
//     const Eigen::Matrix4d& T_ac,
//     UpdateType update_type,
//     FloatingPoint bounding_sphere_radius) const {

//   if (update_type == UpdateType::kFullyUnobserved ||
//       update_type == UpdateType::kFullyFree) {
//     return true;
//   }

//   // compute reprojected image index
//   const auto image_index =
//       projection_model_->imageToNearestIndex(sensor_coordinates.image);

//   // if index is not within bounds, return 0 update
//   if (!depth_image_->isIndexWithinBounds(image_index)) {
//     return 0.f;
//   }

//   // compute T_cp_tilde using the depth image
//   FloatingPoint d_p = depth_image_->at(image_index);

//   // if depth image does not see anything within a reasonable nonzero range, or is non-finite, skip update
//   // as we cannot construct a valid T_cp_tilde
//   if (!std::isfinite(d_p) || d_p <= 1e-3f) {
//     return false;
//   }
  
//   // compute r_pz_c_tilde, the reprojected voxel center on the depth image
//   Point3D r_pz_c_tilde = projection_model_->sensorToCartesian(
//     SensorCoordinates{sensor_coordinates.image, d_p});
  
//   // compute the point defined by the maximal spread from the voxel center
//   Point3D r_pz_c_bounding = r_pcenterz_c + Vector3D::Constant(bounding_sphere_radius / std::sqrt(3.f));

//   // now, compute T_ap and T_ap_tilde using gram-schmidt orthogonalization
//   Eigen::Matrix4d T_cp = compute_gram_schmidt(r_pcenterz_c);
//   Eigen::Matrix4d T_cp_max = compute_gram_schmidt(r_pmaxz_c);
//   Eigen::Matrix4d T_cp_min = compute_gram_schmidt(r_pminz_c);
//   Eigen::Matrix4d T_cp_bounding = compute_gram_schmidt(r_pz_c_bounding);
//   Eigen::Matrix4d T_cp_tilde = compute_gram_schmidt(r_pz_c_tilde);

//   Eigen::Matrix4d T_ap = T_ac * T_cp;
//   Eigen::Matrix4d T_apmax = T_ac * T_cp_max;
//   Eigen::Matrix4d T_apmin = T_ac * T_cp_min;
//   Eigen::Matrix4d T_apbounding = T_ac * T_cp_bounding;
//   Eigen::Matrix4d T_ap_tilde = T_ac * T_cp_tilde;
//   Eigen::Matrix4d T_p_tilde_a = SE3::inverse(T_ap_tilde);

//   // check that both transformations are valid
//   if (!T_apmax.allFinite() || !T_ap_tilde.allFinite()) {
//     LOG(WARNING) << "Invalid transformation matrix T_ap or T_ap_tilde, skipping update.";
//     return 0.f;
//   }

//   const Eigen::Matrix<double, 6, 1> delta_xi_pa =
//     SE3::Log(
//       SE3::inverse(T_ap) * (T_ap_tilde));
//   const Eigen::Matrix<double, 6, 1> delta_x_pa = 
//     SE3::leftJacobianNonScrew(delta_xi_pa) * delta_xi_pa;

//   const Eigen::Matrix<double, 6, 1> delta_xi_pmaxa =
//     SE3::Log(
//       SE3::inverse(T_apmax) * (T_ap));
//   const Eigen::Matrix<double, 6, 1> delta_x_pmaxa = 
//     SE3::leftJacobianNonScrew(delta_xi_pmaxa) * delta_xi_pmaxa;

//   const Eigen::Matrix<double, 6, 1> delta_xi_pbounding =
//     SE3::Log(
//       SE3::inverse(T_apbounding) * (T_ap));
//   const Eigen::Matrix<double, 6, 1> delta_x_pbounding = 
//     SE3::leftJacobianNonScrew(delta_xi_pbounding) * delta_xi_pbounding;

//   const Eigen::Matrix<double, 6, 1> delta_xi_pmina =
//     SE3::Log(
//       SE3::inverse(T_apmin) * (T_ap_tilde));
//   const Eigen::Matrix<double, 6, 1> delta_x_pmina = 
//     SE3::leftJacobianNonScrew(delta_xi_pmina) * delta_xi_pmina;

//   // ensure that delta_xi_ap is finite, non-zero
//   if (!delta_xi_pmaxa.allFinite()) {
//     LOG(WARNING) << "Invalid Lie algebra difference delta_xi_ap, skipping update.";
//     return 0.f;
//   }

//   // get r_pz_c from T_cp_tilde and divide by the depth to get the unit direction
//   Eigen::Vector3d u_pz_c = T_cp_tilde.block<3,1>(0,3);
//   u_pz_c /= d_p;

//   // retrieve the full covariance matrix
//   Eigen::Matrix<double, 6, 6> Eta_x_pa;
//   Eigen::Matrix<double, 6, 6> Eta_x_pa_inv;

//   // generate query pair
//   std::pair<int, int> query_pair(image_index.x(), image_index.y());

//   // check if the covariance matrix for this pixel has already been computed and cached
//   if (covariance_map_.find(query_pair) != covariance_map_.end() && information_map_.find(query_pair) != information_map_.end()) {

//     // if so, retrieve it from the map
//     Eta_x_pa = covariance_map_.at(query_pair);
//     Eta_x_pa_inv = information_map_.at(query_pair);
//   }
//   else {
//     const Eigen::Matrix<double, 6, 6> Eta_xi_ap =
//         getFullCovarianceMatrix(image_index, T_cp_tilde, T_ac, u_pz_c);

//     // use adjoint trick to transform covariance to T_pa frame
//     const Eigen::Matrix<double, 6, 6> Ad_T_pa = 
//         SE3::adjoint(T_p_tilde_a);
//     const Eigen::Matrix<double, 6, 6> Eta_xi_pa = 
//         Ad_T_pa * Eta_xi_ap * Ad_T_pa.transpose();

//     const Eigen::Matrix<double, 6, 6> left_jac_inv_t_pa_tilde_ns = 
//         SE3::leftJacobianInverseNonScrew(SE3::Log(T_p_tilde_a));

//     // now, transform back to cartesian space using the left jacobian at T_pa_tilde
//     Eta_x_pa = left_jac_inv_t_pa_tilde_ns * Eta_xi_pa * left_jac_inv_t_pa_tilde_ns.transpose();

//     // compute the inverse of the covariance matrix
//     Eta_x_pa_inv = Eta_x_pa.inverse();

//     // cache the covariance matrix and information matrix for this pixel. make copies of the matrices to avoid issues with Eigen's lazy evaluation and potential dangling references
//     std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);

//     if (lock.owns_lock()) {
//       covariance_map_.emplace(query_pair, Eta_x_pa);
//       information_map_.emplace(query_pair, Eta_x_pa_inv);
//     }
//   }
  
//   // compute mahalanobis distance to voxel center
//   FloatingPoint mahalanobis_distance_center = std::sqrt(delta_x_pa.transpose() * Eta_x_pa_inv * delta_x_pa);
//   FloatingPoint mahalanobis_distance_bounding = std::sqrt(delta_x_pbounding.transpose() * Eta_x_pa_inv * delta_x_pbounding);

//   // check if the endpoint is contained in the voxel
//   bool voxel_contained = (r_pz_c_tilde - r_pcenterz_c).norm() <= bounding_sphere_radius;
//   bool worst_case_val = false;

//   if (voxel_contained) {
//     // compute the normalized delta_x_pa
//     Eigen::Matrix<double, 6, 1> normalized_delta_x_pa = delta_x_pa / delta_x_pa.norm();

//     // now, compute the maximal displacement to be within the multivariate 3-sigma bounds
//     // as per maximal_delta_x = sqrt(chi2inv(0.9973, 6))**2 / (normalized_delta_x_pa.transpose() * Eta_x_pa_inv * normalized_delta_x_pa)
//     float maximal_delta_x = (4.48f) / std::sqrt(normalized_delta_x_pa.transpose() * Eta_x_pa_inv * normalized_delta_x_pa);

//     const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap = Eta_x_pa.diagonal();
//     double depth_sigma = std::sqrt(diag_Eta_xi_ap(5));

//     float maximal_delta_depth = depth_sigma / 10;

//     worst_case_val = (maximal_delta_depth > delta_x_pmaxa(5)) && (mahalanobis_distance_center < 1.f);
    
//     // if (worst_case_val) {
//     //   LOG(INFO) << "Voxel contained, mahalanobis_distance_center: " << mahalanobis_distance_center << ", maximal_delta_depth: " << maximal_delta_depth << ", delta_x_pmaxa(5): " << delta_x_pmaxa(5) << ", worst_case_val: " << worst_case_val << ", bounding radius: " << bounding_sphere_radius << ", depth_sigma: " << depth_sigma << ", normalized_delta_x_pa: " << normalized_delta_x_pa.transpose();
//     // }
//   }

//   // if both are within multivariate 3-sigma bounds, update as the approximation error is acceptable
//   return worst_case_val;
// }

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

// void LieContinuousRay::generateCovarianceAndInformationMatrices(
//     const PosedImage<>::Ptr posed_depth_image,
//     const Image<FloatingPoint>::Ptr depth_uncertainty_image,
//     std::shared_ptr<MatrixT<double>> pose_covariance) const {

//     // iterate through every element of the pose_depth_image, and for all elements within the min/max range, compute the covariance and information matrices and store them in the internal maps
//     for (int y = 0; y < posed_depth_image->getNumR(); ++y) {
//       for (int x = 0; x < posed_depth_image->getWidth(); ++x) {
//         const Index2D image_index(x, y);

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

  // compute cell width from the euclidean norm of r_pmaxz_c - r_pcenterz_c
  const FloatingPoint cell_width = (r_pmaxz_c - r_pcenterz_c).norm();

  // compute T_cp_tilde using the depth image
  FloatingPoint d_p = depth_image_->at(image_index);

  // if depth is < 0, it means that the center pixel projects to something invalid, in that case, we need to define a pixel region over the depth image, and iterate through that region to see if there is a valid pixel anywhere

  if (d_p < 0.f) {
    
    // compute the image coordinates of the voxel max corner and take the norm as the search radius from the center
    const auto image_coordinates_max = projection_model_->cartesianToImage(r_pmaxz_c);
    const auto image_coordinates_center = projection_model_->cartesianToImage(r_pcenterz_c);
    const FloatingPoint search_radius = (image_coordinates_max - image_coordinates_center).norm();

    // compute the bounding box of the search region
    const int x_min = std::max(0, static_cast<int>(std::floor(image_coordinates_center.x() - search_radius)));
    const int x_max = std::min(depth_image_->getNumColumns() - 1, static_cast<int>(std::ceil(image_coordinates_center.x() + search_radius)));
    const int y_min = std::max(0, static_cast<int>(std::floor(image_coordinates_center.y() - search_radius)));
    const int y_max = std::min(depth_image_->getNumRows() - 1, static_cast<int>(std::ceil(image_coordinates_center.y() + search_radius)));

    // iterate through the bounding box and find the closest valid depth
    FloatingPoint closest_depth = std::numeric_limits<FloatingPoint>::infinity();
    for (int y = y_min; y <= y_max; ++y) {
      for (int x = x_min; x <= x_max; ++x) {
        const FloatingPoint depth = depth_image_->at(Index2D(x, y));
        if (std::isfinite(depth) && depth > 1e-3f && depth < closest_depth) {
          closest_depth = depth;
          break;
        }
      }
      if (closest_depth < std::numeric_limits<FloatingPoint>::infinity()) {
        break;
      }
    }


    // if we found a valid depth, use it
    if (closest_depth < std::numeric_limits<FloatingPoint>::infinity()) {
      d_p = closest_depth;
    }
    else {
      // if we didn't find a valid depth, return 0 update
      return 0.f;
    }
  }


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
    // SE3::Log(
    //   T_p_tilde_a * (T_ap));
    SE3::Log(
      SE3::inverse(T_ap) * (T_ap_tilde));
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
  // Eigen::MatrixXd Eta_x_pa;
  // Eigen::MatrixXd Eta_x_pa_inv;
  Eigen::Matrix<double, 6, 6> Eta_x_pa;
  Eigen::Matrix<double, 6, 6> Eta_x_pa_inv;

  // generate query pair
  std::pair<int, int> query_pair(image_index.x(), image_index.y());

  // check if the covariance matrix for this pixel has already been computed and cached
  if (covariance_map_.find(query_pair) != covariance_map_.end() && information_map_.find(query_pair) != information_map_.end()) {

    // if so, retrieve it from the map
    Eta_x_pa = covariance_map_.at(query_pair);
    Eta_x_pa_inv = information_map_.at(query_pair);
  }
  else {
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
    Eta_x_pa = left_jac_inv_t_pa_tilde_ns * Eta_xi_pa * left_jac_inv_t_pa_tilde_ns.transpose();

    // compute the inverse of the covariance matrix
    Eta_x_pa_inv = Eta_x_pa.inverse();

    // cache the covariance matrix and information matrix for this pixel. make copies of the matrices to avoid issues with Eigen's lazy evaluation and potential dangling references
    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);

    if (lock.owns_lock()) {
      covariance_map_.emplace(query_pair, Eta_x_pa);
      information_map_.emplace(query_pair, Eta_x_pa_inv);
    }
  }

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

  // -------------------- NEW UPDATE STRUCTURE ---------------------
  //if so, compute the full beam update

  // now, check if we are within the bounds for every dimension aside from depth
  FloatingPoint mahalanobis_distance_no_depth = std::sqrt(
    delta_x_pa.head<5>().transpose() *
    Eta_x_pa_inv.topLeftCorner<5,5>() *
    delta_x_pa.head<5>());

  // first, check if we are behind the mean endpoint, if so, return 0 update

  // first, check if delta_x_pa is within the bounds of the endpoint uncertainty
   if (true) {  // 4.48 is the chi2inv(0.9973, 6) value
    
    // now, see if we are within the depth bounds of the endpoint uncertainty, 3-sigma in front, and 6-sigma behind
    const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap = Eta_x_pa.diagonal();
    double depth_sigma = std::sqrt(diag_Eta_xi_ap(5));
    if (delta_x_pa(5) > 0) {
      if (3 * depth_sigma >= delta_x_pa(5)) {
        return computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);
      }
      else {
        return computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);
      }
    }
    else {
      if (depth_sigma >= -delta_x_pa(5)) {
        return computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);
      }
      else {
        return 0.f;
      }
    }
  }
  else {
    LOG(INFO) << "r_pc_z_tilde: " << r_pz_c_tilde.transpose() << ", r_pcenterz_c: " << r_pcenterz_c.transpose() << ", delta_x_pa(5): " << delta_x_pa(5);
    // else, we are outside the bounds, return 0 update
    return 0.f;
  }
  // -------------------- NEW UPDATE STRUCTURE ENDS ---------------------


  // -------------------- OLD UPDATE STRUCTURE ---------------------
  // //if so, compute the full beam update

  // // calculate the mahalanobis distance
  // FloatingPoint mahalanobis_distance = std::sqrt(delta_x_pa.transpose() * Eta_x_pa_inv * delta_x_pa);

  // // first, check if we are behind the mean endpoint, if so, return 0 update

  // // first, check if delta_x_pa is within the bounds of the endpoint uncertainty
  // if ((mahalanobis_distance <= 4.48f)) {  // 4.48 is the chi2inv(0.9973, 6) value
  //   // now, retrieve diagonal elements of the full covariance as a vector
  //   const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap = Eta_x_pa.diagonal();
  //   double f_depth = -(delta_x_pa(5)) / std::sqrt(diag_Eta_xi_ap(5));
  //   // LOG(INFO) << "Delta_x_pa is within the bounds of the endpoint uncertainty with delta_x_pa(5) of " << f_depth << ", computing full beam update.";
  //   //if so, compute the full beam update
  //   FloatingPoint sample_update = computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);

  //   return sample_update;
  // }
  // else {

  //   // now, check if we are within the bounds for every dimension aside from depth
  //   FloatingPoint mahalanobis_distance_no_depth = std::sqrt(
  //     delta_x_pa.head<5>().transpose() *
  //     Eta_x_pa_inv.topLeftCorner<5,5>() *
  //     delta_x_pa.head<5>());

  //   if (mahalanobis_distance_no_depth <= 3.65f) {  // 3.65 is the chi2inv(0.9973, 5) value
  //     // we are within bounds for all but depth, return free space update if delta_x_pa(5) < 0
  //     if (delta_x_pa(5) > 0.f) {
  //       // return computeFreeSpaceBeamUpdate();
  //       return computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);
  //     }
  //     else if (delta_x_pa(5) > -10 * cell_width) {
  //       // now, retrieve diagonal elements of the full covariance as a vector
  //       const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap = Eta_x_pa.diagonal();
  //       double f_depth = -(delta_x_pa(5)) / std::sqrt(diag_Eta_xi_ap(5));
  //       // LOG(INFO) << "Delta_x_pa(5) is negative but within 10 cell widths with exact scaled value of " << f_depth << ".";
  //       return computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);
  //     }
  //   }

  //   // else, we are outside the bounds or behind the surface, return 0 update
  //   return 0.f;
  // }
  // -------------------- OLD UPDATE STRUCTURE ENDS ---------------------

  // TODO: light refactor, need to check

  // check if we are within the non-depth bounds of the endpoint uncertainty
  // FloatingPoint mahalanobis_distance_no_depth = std::sqrt(
  //   delta_x_pa.head<5>().transpose() *
  //   Eta_x_pa_inv.topLeftCorner<5,5>() *
  //   delta_x_pa.head<5>());

  // if (mahalanobis_distance_no_depth <= 3.65f) {  // 3.65 is the chi2inv(0.9973, 5) value
  //     if (delta_x_pa(5) > 0.f) {
  //       // return computeFreeSpaceBeamUpdate();
  //       return computeFullBeamUpdate(delta_x_pa, Eta_x_pa, d_p, cell_width);
  //     }
  // }

  return 0.f;
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

  // add 1e-3 to the diagonal of the pose covariance to ensure positive definiteness
  const Eigen::Matrix<double, 6, 6> synth_pos_cov = *pose_covariance_;

  // then, compute the total uncertainty
  // const Eigen::Matrix<double, 6, 6> Eta_xi_ap = Eta_ac + (sigma_d * sigma_d) * J_left * J_left_inv_non_screw * u_pz_c_6d * u_pz_c_6d.transpose() * J_left_inv_non_screw.transpose() * J_left.transpose();
  const Eigen::Matrix<double, 6, 6> Eta_xi_ap =
      synth_pos_cov + Ad_T_ac * J_left * J_left_inv_non_screw * u_pz_c_6d * u_pz_c_6d.transpose() * J_left_inv_non_screw.transpose() * J_left.transpose() * Ad_T_ac.transpose() * (sigma_d * sigma_d);

  // throw an error if covariance matrix is not positive definite
  // Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt_of_Eta_xi_ap(Eta_xi_ap);
  // DCHECK_EQ(llt_of_Eta_xi_ap.info(), Eigen::Success)
  //     << "Covariance matrix is not positive definite!";

  return Eta_xi_ap;
}

FloatingPoint LieContinuousRay::computeFullBeamUpdate(
    const Eigen::Matrix<double, 6, 1>& delta_xi_ap,
    const Eigen::Matrix<double, 6, 6>& Eta_xi_ap,
    FloatingPoint d_p,
    FloatingPoint cell_width) const {

  // now, retrieve diagonal elements of the full covariance as a vector
  const Eigen::Matrix<double, 1, 6> diag_Eta_xi_ap =
      Eta_xi_ap.diagonal();

  double support_fac = 1.0;

  // based on a maximum depth of voxel_res * 12, as per supereight2
  const double max_depth_sigma = cell_width * 12;

  const double depth_sigma = std::sqrt(diag_Eta_xi_ap(5));
  support_fac = std::min(1.0, max_depth_sigma / depth_sigma);

  // prevent support_fac from being too small
  support_fac = std::max(.3, support_fac);

  // calculate the depth contribution using the last element of the diagonal
  const double f_depth = -(delta_xi_ap(5)) / std::sqrt(diag_Eta_xi_ap(5));

  // ensure that f_depth is finite and non-zero
  DCHECK(std::isfinite(f_depth) && f_depth != 0.0);
  
  double scaled_contrib =
      ApproximateGaussianDistribution::cumulative(f_depth) -
      0.5f * ApproximateGaussianDistribution::cumulative(f_depth - 3.f * support_fac) - 0.5f;

  // now, iterate through the other elements to compute the other contributions using C(g + 3.f) - C(g - 3.f)
  for (int i = 0; i < 5; ++i) {
    const double f_other =
        std::abs(delta_xi_ap(i)) / std::sqrt(diag_Eta_xi_ap(i));
    
    double support_fac = 1.0;

    // if i is {0, 1, 2}, then define the maximum angular sigma as 5 degrees
    if (i < 3) {
      // const double max_angular_sigma = .1f * M_PI / 180.f; // 5 degrees in radians

      // compute the maximum angular sigma from the depth based on a desired opposite edge length of 0.01 meters, i.e. max_angular_sigma = atan(0.01 / d_p)
      const double max_angular_sigma = std::atan2(0.01, d_p);
      const double angle_sigma = std::sqrt(diag_Eta_xi_ap(i));

      support_fac = std::min(1.0, max_angular_sigma / angle_sigma);
      support_fac = std::max(.3, support_fac);

      scaled_contrib *=
          (ApproximateGaussianDistribution::cumulative(f_other + 3.f * support_fac) -
          ApproximateGaussianDistribution::cumulative(f_other - 3.f * support_fac));
    }
    // we define the maximum x, y sigma as .05 meters
    else {
      const double max_xy_sigma = 0.05f;
      const double xy_sigma = std::sqrt(diag_Eta_xi_ap(i));
      support_fac = std::min(1.0, max_xy_sigma / xy_sigma);
      support_fac = std::max(.3, support_fac);

      scaled_contrib *=
          (ApproximateGaussianDistribution::cumulative(f_other + 3.f * support_fac) -
          ApproximateGaussianDistribution::cumulative(f_other - 3.f * support_fac));
    }
  }

  const double p = scaled_contrib + 0.5f;

  // if the update is a free-space update, then we scale by a proportional amount to the support factors to avoid sending a complete free-space update when the support factors are small, which would be an overconfident update
  if (p < 0.5f) {
    double support_fac_product = 1.0;
    for (int i = 0; i < 6; ++i) {
      const double sigma_i = std::sqrt(diag_Eta_xi_ap(i));
      if (i < 3) {
        const double max_angular_sigma = std::atan2(0.01, d_p);
        support_fac_product *= std::min(1.0, max_angular_sigma / sigma_i);
      }
      else if (i < 5) {
        const double max_xy_sigma = 0.05f;
        support_fac_product *= std::min(1.0, max_xy_sigma / sigma_i);
      }
      else {
        const double max_depth_sigma = cell_width * 12;
        support_fac_product *= std::min(1.0, max_depth_sigma / sigma_i);
      }
    }
    const double p_scaled = scaled_contrib * support_fac_product + 0.5f;
    // scale the update by the product of the support factors
    return std::log((p_scaled) / (1.f - p_scaled));
  }
  // if (std::abs(delta_xi_ap(5)) < cell_width) {
  //   double depth_scaled_contrib = ApproximateGaussianDistribution::cumulative(f_depth) - 0.5f * ApproximateGaussianDistribution::cumulative(f_depth - 3.f * support_fac) - 0.5f;
  //   LOG(INFO) << "Delta_xi_ap(5) is less than cell width of " << cell_width << ", and the update is " << p << ", with f_depth of " << f_depth << " and depth_sigma of " << depth_sigma << " and support_fac of " << support_fac << " and depth_scaled_contrib of " << depth_scaled_contrib;
  // }
  const double log_odds = std::log(p / (1.f - p));
  DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
  return log_odds;
}

FloatingPoint LieContinuousRay::computeFreeSpaceBeamUpdate() const {
  // const FloatingPoint p = -0.5f * config_.scaling_free + 0.5f;
  const FloatingPoint p = 0.0001f; // a small free space update
  const FloatingPoint log_odds = std::log(p / (1.f - p));
  DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
  return log_odds;
}

}  // namespace wavemap
