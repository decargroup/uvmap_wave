#ifndef WAVEMAP_CORE_INTEGRATOR_MEASUREMENT_MODEL_IMPL_CONTINUOUS_RAY_INL_H_
#define WAVEMAP_CORE_INTEGRATOR_MEASUREMENT_MODEL_IMPL_CONTINUOUS_RAY_INL_H_

#include "wavemap/core/integrator/measurement_model/approximate_gaussian_distribution.h"
#include "wavemap/3rd_party/pymlg/se3.h"

namespace wavemap {
inline FloatingPoint ContinuousRay::computeWorstCaseApproximationError(
    UpdateType update_type, FloatingPoint /*sphere_center_distance*/,
    FloatingPoint cell_bounding_radius) const {
  if (update_type == UpdateType::kFullyUnobserved ||
      update_type == UpdateType::kFullyFree) {
    return 0.f;
  }

  if (update_type == UpdateType::kFreeOrUnobserved) {
    // TODO(victorr): Check this
    return 0.5f * config_.scaling_free;
  }

  const FloatingPoint worst_dz = cell_bounding_radius;
  const FloatingPoint worst_dz_error =
      3.f / 8.f * config_.scaling_occupied / config_.range_sigma * worst_dz;

  return worst_dz_error;
}

inline FloatingPoint ContinuousRay::computeUpdate(
    const SensorCoordinates& sensor_coordinates) const {
  switch (config_.beam_selector_type) {
    case BeamSelectorType::kNearestNeighbor: {
      const auto image_index =
          projection_model_->imageToNearestIndex(sensor_coordinates.image);
      return computeBeamUpdate(sensor_coordinates, image_index);
    }
    case BeamSelectorType::kAllNeighbors: {
      FloatingPoint update = 0.f;
      const auto image_indices =
          projection_model_->imageToNearestIndices(sensor_coordinates.image);
      for (int neighbor_idx = 0; neighbor_idx < 4; ++neighbor_idx) {
        const Index2D& image_index = image_indices.col(neighbor_idx);
        update += computeBeamUpdate(sensor_coordinates, image_index);
      }
      return update;
    }
    default:
      return 0.f;
  }
}

inline FloatingPoint ContinuousRay::computeBeamUpdate(
    const SensorCoordinates& sensor_coordinates,
    const Index2D& image_index) const {
  if (!range_image_->isIndexWithinBounds(image_index)) {
    return 0.f;
  }

  const FloatingPoint measured_distance = range_image_->at(image_index);

  if (range_threshold_back_ < sensor_coordinates.depth - measured_distance) {
    return 0.f;
  }

  if (sensor_coordinates.depth < measured_distance - range_threshold_front_) {
    return computeFreeSpaceBeamUpdate();
  } else {
    return computeFullBeamUpdate(sensor_coordinates.depth, measured_distance);
  }
}

inline FloatingPoint ContinuousRay::computeFullBeamUpdate(
    FloatingPoint cell_to_sensor_distance,
    FloatingPoint measured_distance) const {
  const FloatingPoint f =
      (cell_to_sensor_distance - measured_distance) / config_.range_sigma;
  const FloatingPoint range_contrib =
      ApproximateGaussianDistribution::cumulative(f) -
      0.5f * ApproximateGaussianDistribution::cumulative(f - 3.f) - 0.5f;

  const FloatingPoint scaled_contribs =
      (range_contrib < 0.f) ? config_.scaling_free * range_contrib
                            : config_.scaling_occupied * range_contrib;

  const FloatingPoint p = scaled_contribs + 0.5f;
  const FloatingPoint log_odds = std::log(p / (1.f - p));
  DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
  return log_odds;
}

inline FloatingPoint ContinuousRay::computeFreeSpaceBeamUpdate() const {
  const FloatingPoint p = -0.5f * config_.scaling_free + 0.5f;
  const FloatingPoint log_odds = std::log(p / (1.f - p));
  DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
  return log_odds;
}

// ---------------------------------------------------------------------------
// inline FloatingPoint LieContinuousRay::computeWorstCaseApproximationError(
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
//       3.f / 8.f * config_.scaling_occupied / config_.range_sigma * worst_dz;

//   return worst_dz_error;
// }

// inline FloatingPoint LieContinuousRay::computeUpdate(
//     const Eigen::Matrix<FloatingPoint, 6, 1>& delta_xi_ap,
//     const TransformationMatrix& T_cp_tilde,
//     const SensorCoordinates& sensor_coordinates) const {
//   switch (config_.beam_selector_type) {
//     case BeamSelectorType::kNearestNeighbor: {
//       const auto image_index =
//           projection_model_->imageToNearestIndex(sensor_coordinates.image);
//       return computeBeamUpdate(delta_xi_ap, image_index, T_cp_tilde);
//     }
//     case BeamSelectorType::kAllNeighbors: {
//       FloatingPoint update = 0.f;
//       const auto image_indices =
//           projection_model_->imageToNearestIndices(sensor_coordinates.image);
//       for (int neighbor_idx = 0; neighbor_idx < 4; ++neighbor_idx) {
//         const Index2D& image_index = image_indices.col(neighbor_idx);
//         update += computeBeamUpdate(delta_xi_ap, image_index, T_cp_tilde);
//       }
//       return update;
//     }
//     default:
//       return 0.f;
//   }
// }

// inline FloatingPoint LieContinuousRay::computeBeamUpdate(
//     const Eigen::Matrix<FloatingPoint, 6, 1>& delta_xi_ap,
//     const Index2D& image_index,
//     const TransformationMatrix& T_cp_tilde) const {
//   if (!depth_image_->isIndexWithinBounds(image_index)) {
//     return 0.f;
//   }

//   // retrieve the full covariance matrix
//   const Eigen::Matrix<FloatingPoint, 6, 6> Eta_xi_ap =
//       getFullCovarianceMatrix(delta_xi_ap, image_index, T_cp_tilde);

//   const FloatingPoint measured_distance = depth_image_->at(image_index);

//   // if (range_threshold_back_ < sensor_coordinates.depth - measured_distance) {
//   //   return 0.f;
//   // }
//   if ((3 * Eta_xi_ap(5,5)) < (delta_xi_ap(5) * delta_xi_ap(5))) {
//     return 0.f;
//   }

//   // if (sensor_coordinates.depth < measured_distance - range_threshold_front_) {
//   //   return computeFreeSpaceBeamUpdate();
//   // } else {
//   //   return computeFullBeamUpdate(delta_xi_ap, Eta_xi_ap);
//   // }
//   if ((delta_xi_ap(5) * delta_xi_ap(5)) < -(3 * Eta_xi_ap(5,5))) {
//     return computeFreeSpaceBeamUpdate();
//   } else {
//     return computeFullBeamUpdate(delta_xi_ap, Eta_xi_ap);
//   }
// }

// inline Eigen::Matrix<FloatingPoint, 6, 6> LieContinuousRay::getFullCovarianceMatrix(const Eigen::Matrix<FloatingPoint, 6, 1>& delta_xi_ap, const Index2D& image_index, const TransformationMatrix& T_cp_tilde) const {
//   // first, get the depth uncertainty at image_index
//   const FloatingPoint depth_uncertainty = depth_uncertainty_image_->at(image_index);

//   // now, define a dilation matrix for the depth uncertainty
//   Eigen::Matrix<FloatingPoint, 6, 1> L = Eigen::Matrix<FloatingPoint, 6, 1>::Zero();
//   L(5) = 1.f;

//   // and the inverse left jacobian to map it to the Lie algebra
//   // Eigen::Matrix<FloatingPoint, 6, 6> J_inv =
//   //     SE3::leftJacobianInverse(
//   //       SE3::log(T_cp_tilde));

//   // then, compute the total uncertainty
//   // const Eigen::Matrix<FloatingPoint, 6, 6> Eta_xi_ap = (depth_uncertainty * depth_uncertainty) * J_inv * L * L.transpose() * J_inv.transpose();
//   const Eigen::Matrix<FloatingPoint, 6, 6> Eta_xi_ap = L * L.transpose();
//       // pose_covariance_ + (depth_uncertainty * depth_uncertainty) * J_inv * L * L.transpose() * J_inv.transpose();

//   return Eta_xi_ap;
// }

// inline FloatingPoint LieContinuousRay::computeFullBeamUpdate(
//     const Eigen::Matrix<FloatingPoint, 6, 1>& delta_xi_ap,
//     const Eigen::Matrix<FloatingPoint, 6, 6>& Eta_xi_ap) const {

//   // now, retrieve diagonal elements of the full covariance as a vector
//   const Eigen::Matrix<FloatingPoint, 1, 6> diag_Eta_xi_ap =
//       Eta_xi_ap.diagonal();

//   // calculate the depth contribution using the last element of the diagonal
//   const FloatingPoint f_depth =
//       (delta_xi_ap(5)) / std::sqrt(diag_Eta_xi_ap(5));
//   FloatingPoint scaled_contrib =
//       ApproximateGaussianDistribution::cumulative(f_depth) -
//       0.5f * ApproximateGaussianDistribution::cumulative(f_depth - 3.f) - 0.5f;

//   // now, iterate through the other elements to compute the other contributions using C(g + 3.f) - C(g - 3.f)
//   for (int i = 0; i < 5; ++i) {
//     const FloatingPoint f_other =
//         (delta_xi_ap(i)) / std::sqrt(diag_Eta_xi_ap(i));
//     scaled_contrib *=
//         (ApproximateGaussianDistribution::cumulative(f_other + 3.f) -
//          ApproximateGaussianDistribution::cumulative(f_other - 3.f));
//   }

//   const FloatingPoint p = scaled_contrib + 0.5f;
//   const FloatingPoint log_odds = std::log(p / (1.f - p));
//   DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
//   return log_odds;
// }

// inline FloatingPoint LieContinuousRay::computeFreeSpaceBeamUpdate() const {
//   const FloatingPoint p = -0.5f * config_.scaling_free + 0.5f;
//   const FloatingPoint log_odds = std::log(p / (1.f - p));
//   DCHECK(!std::isnan(log_odds) && std::isfinite(log_odds));
//   return log_odds;
// }

}  // namespace wavemap

#endif  // WAVEMAP_CORE_INTEGRATOR_MEASUREMENT_MODEL_IMPL_CONTINUOUS_RAY_INL_H_
