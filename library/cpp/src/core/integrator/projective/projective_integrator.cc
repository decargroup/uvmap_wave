#include "wavemap/core/integrator/projective/projective_integrator.h"

#include <wavemap/core/utils/data/eigen_checks.h>
#include <wavemap/core/utils/profile/profiler_interface.h>

namespace wavemap {
DECLARE_CONFIG_MEMBERS(ProjectiveIntegratorConfig,
                      (min_range)
                      (max_range)
                      (max_update_resolution)
                      (termination_update_error));

bool ProjectiveIntegratorConfig::isValid(bool verbose) const {
  bool is_valid = true;

  is_valid &= IS_PARAM_GT(min_range, 0.f, verbose);
  is_valid &= IS_PARAM_GT(max_range, 0.f, verbose);
  is_valid &= IS_PARAM_LT(min_range, max_range, verbose);
  is_valid &= IS_PARAM_GE(max_update_resolution, 0.f, verbose);
  is_valid &= IS_PARAM_GT(termination_update_error, 0.f, verbose);

  return is_valid;
}

void ProjectiveIntegrator::integrate(const PosedPointcloud<>& pointcloud) {
  ProfilerZoneScoped;
  if (!isPoseValid(pointcloud.getPose())) {
    return;
  }
  importPointcloud(pointcloud);
  updateMap();
}

void ProjectiveIntegrator::integrate(const PosedImage<>& range_image) {
  ProfilerZoneScoped;
  CHECK_NOTNULL(projection_model_);
  if (range_image.getDimensions() != projection_model_->getDimensions()) {
    LOG(WARNING) << "Dimensions of range image"
                 << print::eigen::oneLine(range_image.getDimensions())
                 << " do not match projection model"
                 << print::eigen::oneLine(projection_model_->getDimensions())
                 << ". Ignoring integration request.";
    return;
  }
  if (!isPoseValid(range_image.getPose())) {
    return;
  }
  importRangeImage(range_image);
  updateMap();
}

void ProjectiveIntegrator::importPointcloud(
    const PosedPointcloud<>& pointcloud) {
  ProfilerZoneScoped;
  // Reset the posed range image and the beam offset image
  posed_range_image_->resetToInitialValue();
  posed_range_image_->setPose(pointcloud.getPose());
  beam_offset_image_->resetToInitialValue();

  // Import all the points
  for (const auto& C_point : pointcloud.getPointsLocal()) {
    // Filter out noisy points and compute point's range
    if (!isMeasurementValid(C_point)) {
      continue;
    }

    // Calculate the range image index
    const auto sensor_coordinates =
        projection_model_->cartesianToSensor(C_point);

    const auto [range_image_index, beam_to_pixel_offset] =
        projection_model_->imageToNearestIndexAndOffset(
            sensor_coordinates.image);
    if (!posed_range_image_->isIndexWithinBounds(range_image_index)) {
      // Prevent out-of-bounds access
      continue;
    }

    // Add the point to the range image, if multiple points hit the same image
    // pixel, keep the closest point
    const FloatingPoint range = sensor_coordinates.depth;
    const FloatingPoint old_range_value =
        posed_range_image_->at(range_image_index);
    if (old_range_value < config_.min_range || range < old_range_value) {
      posed_range_image_->at(range_image_index) = range;
      beam_offset_image_->at(range_image_index) = beam_to_pixel_offset;
    }
  }
}

void ProjectiveIntegrator::importRangeImage(
    const PosedImage<>& range_image_input) {
  ProfilerZoneScoped;
  CHECK_NOTNULL(posed_range_image_);
  CHECK_EIGEN_EQ(range_image_input.getDimensions(),
                 posed_range_image_->getDimensions());
  *posed_range_image_ = range_image_input;
  beam_offset_image_->resetToInitialValue();
}

//  -------------------------------- LieProjectiveIntegrator --------------------------------


void LieProjectiveIntegrator::integrate(const PosedImage<>& range_image, const Image<FloatingPoint>& depth_uncertainty_image, const MatrixT<double>& pose_covariance) {
  ProfilerZoneScoped;
  CHECK_NOTNULL(projection_model_);
  if (range_image.getDimensions() != projection_model_->getDimensions()) {
    LOG(WARNING) << "Dimensions of range image"
                 << print::eigen::oneLine(range_image.getDimensions())
                 << " do not match projection model"
                 << print::eigen::oneLine(projection_model_->getDimensions())
                 << ". Ignoring integration request.";
    return;
  }
  if (depth_uncertainty_image.getDimensions() !=
      projection_model_->getDimensions()) {
    LOG(WARNING)
        << "Dimensions of depth uncertainty image"
        << print::eigen::oneLine(depth_uncertainty_image.getDimensions())
        << " do not match projection model"
        << print::eigen::oneLine(projection_model_->getDimensions())
        << ". Ignoring integration request.";
    return;
  }

  // check if input covariance matrix is 6x6
  if (pose_covariance.rows() != 6 || pose_covariance.cols() != 6) {
    LOG(WARNING) << "Pose covariance matrix must be 6x6. Ignoring integration request.";
    return;
  }

  // check if input covariance is positive definite
  // Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt_of_Eta_xi_ap(pose_covariance);
  // DCHECK_EQ(llt_of_Eta_xi_ap.info(), Eigen::Success)
  //     << "Covariance matrix is not positive definite!";

  if (!isPoseValid(range_image.getPose())) {
    LOG(WARNING) << "Invalid pose for range image. Ignoring integration request.";
    return;
  }
  // LOG(INFO) << "Importing depth image and associated data into LieProjectiveIntegrator.";
  importDepthImage(range_image);
  importDepthUncertaintyImage(depth_uncertainty_image);
  importPoseCovariance(pose_covariance);
  // LOG(INFO) << "All data imported into LieProjectiveIntegrator, updating map.";
  updateMap();
}

void LieProjectiveIntegrator::importDepthImage(
    const PosedImage<>& depth_image_input) {
  ProfilerZoneScoped;
  CHECK_NOTNULL(posed_depth_image_);
  CHECK_EIGEN_EQ(depth_image_input.getDimensions(),
                 posed_depth_image_->getDimensions());
  *posed_depth_image_ = depth_image_input;
}

void LieProjectiveIntegrator::importDepthUncertaintyImage(
    const Image<FloatingPoint>& depth_uncertainty_image_input) {
  ProfilerZoneScoped;
  CHECK_NOTNULL(depth_uncertainty_image_);
  CHECK_EIGEN_EQ(depth_uncertainty_image_input.getDimensions(),
                 depth_uncertainty_image_->getDimensions());
  *depth_uncertainty_image_ = depth_uncertainty_image_input;
}

void LieProjectiveIntegrator::importPoseCovariance(
    const MatrixT<double>& pose_covariance_input) {
  ProfilerZoneScoped;
  // CHECK_NOTNULL(pose_covariance_);
  // CHECK_EIGEN_EQ(pose_covariance_input, pose_covariance_);
  *pose_covariance_ = pose_covariance_input;
}

FloatingPoint LieProjectiveIntegrator::computeUpdate(
    const Point3D& r_pcenterz_c,
    const Point3D& r_pmaxz_c,
    const Eigen::Matrix4d& T_ac) const {

  // compute sensor coordinates of the point
  const auto sensor_coordinates =
      projection_model_->cartesianToSensor(r_pcenterz_c);

  // Check if we're outside the min/max range, this will also make sure that the depth is valid
  if (sensor_coordinates.depth < config_.min_range ||
      config_.max_range < sensor_coordinates.depth) {
    return 0.f;
  }

  // return measurement_model_->computeUpdate(delta_xi_ap, T_cp_tilde, sensor_coordinates);
  return measurement_model_->computeUpdate(sensor_coordinates, r_pcenterz_c, r_pmaxz_c, T_ac);
}

}  // namespace wavemap
