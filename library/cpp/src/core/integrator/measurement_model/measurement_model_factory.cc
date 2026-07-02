#include "wavemap/core/integrator/measurement_model/measurement_model_factory.h"

#include <memory>
#include <utility>

#include "wavemap/core/integrator/measurement_model/continuous_beam.h"
#include "wavemap/core/integrator/measurement_model/continuous_ray.h"

namespace wavemap {
std::unique_ptr<MeasurementModelBase> wavemap::MeasurementModelFactory::create(
    const param::Value& params, ProjectorBase::ConstPtr projection_model,
    Image<>::ConstPtr range_image, Image<Vector2D>::ConstPtr beam_offset_image,
    std::optional<MeasurementModelType> default_measurement_model_type) {
  if (const auto type = MeasurementModelType::from(params, "measurement_model");
      type) {
    return create(type.value(), std::move(projection_model),
                  std::move(range_image), std::move(beam_offset_image), params);
  }

  if (default_measurement_model_type.has_value()) {
    LOG(WARNING) << "Default type \""
                 << default_measurement_model_type.value().toStr()
                 << "\" will be created instead.";
    return create(default_measurement_model_type.value(),
                  std::move(projection_model), std::move(range_image),
                  std::move(beam_offset_image), params);
  }

  LOG(ERROR) << "No default was set. Returning nullptr.";
  return nullptr;
}

std::unique_ptr<MeasurementModelBase> wavemap::MeasurementModelFactory::create(
    MeasurementModelType measurement_model_type,
    ProjectorBase::ConstPtr projection_model, Image<>::ConstPtr range_image,
    Image<Vector2D>::ConstPtr beam_offset_image, const param::Value& params) {
  switch (measurement_model_type) {
    case MeasurementModelType::kContinuousRay: {
      if (const auto config =
              ContinuousRayConfig::from(params, "measurement_model");
          config) {
        return std::make_unique<ContinuousRay>(config.value(),
                                               std::move(projection_model),
                                               std::move(range_image));
      } else {
        LOG(ERROR)
            << "Continuous ray measurement model config could not be loaded.";
        return nullptr;
      }
    }
    case MeasurementModelType::kContinuousBeam: {
      if (const auto config =
              ContinuousBeamConfig::from(params, "measurement_model");
          config) {
        return std::make_unique<ContinuousBeam>(
            config.value(), std::move(projection_model), std::move(range_image),
            std::move(beam_offset_image));
      } else {
        LOG(ERROR)
            << "Continuous beam measurement model config could not be loaded.";
        return nullptr;
      }
    }
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
std::unique_ptr<MeasurementModelLieBase> wavemap::MeasurementModelLieFactory::create(
    const param::Value& params, ProjectorBase::ConstPtr projection_model,
    Image<>::ConstPtr depth_image, 
    Image<FloatingPoint>::ConstPtr depth_uncertainty_image,
    std::shared_ptr<const MatrixT<double>> pose_covariance,
    std::optional<MeasurementModelType> default_measurement_model_type) {
  if (const auto type = MeasurementModelType::from(params, "measurement_model");
      type) {
    return create(type.value(), std::move(projection_model),
                  std::move(depth_image), std::move(depth_uncertainty_image), std::move(pose_covariance), params);
  }

  if (default_measurement_model_type.has_value()) {
    LOG(WARNING) << "Default type \""
                 << default_measurement_model_type.value().toStr()
                 << "\" will be created instead.";
    return create(default_measurement_model_type.value(), std::move(projection_model),
                  std::move(depth_image), std::move(depth_uncertainty_image), std::move(pose_covariance), params);
  }

  LOG(ERROR) << "No default was set. Returning nullptr.";
  return nullptr;
}

std::unique_ptr<MeasurementModelLieBase> wavemap::MeasurementModelLieFactory::create(
    MeasurementModelType measurement_model_type,
    ProjectorBase::ConstPtr projection_model, Image<>::ConstPtr depth_image,
    Image<FloatingPoint>::ConstPtr depth_uncertainty_image,
    std::shared_ptr<const MatrixT<double>> pose_covariance, const param::Value& params) {
  switch (measurement_model_type) {
    case MeasurementModelType::kContinuousLieRay: {
      if (const auto config =
              ContinuousRayConfig::from(params, "measurement_model");
          config) {
        return std::make_unique<LieContinuousRay>(config.value(),
                                               std::move(projection_model),
                                               std::move(depth_image),
                                               std::move(depth_uncertainty_image),
                                               std::move(pose_covariance));
      } else {
        LOG(ERROR)
            << "Continuous ray measurement model config could not be loaded.";
        return nullptr;
      }
    }
  }

  LOG(ERROR) << "Unsupported measurement model type for Lie groups, currently only uses LieContinuousRay.";
  return nullptr;
}
}  // namespace wavemap
