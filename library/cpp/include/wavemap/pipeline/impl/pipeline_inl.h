#ifndef WAVEMAP_PIPELINE_IMPL_PIPELINE_INL_H_
#define WAVEMAP_PIPELINE_IMPL_PIPELINE_INL_H_

#include <string>
#include <vector>

namespace wavemap {

template <typename MeasurementT>
bool Pipeline::runIntegrators(const std::vector<std::string>& integrator_names,
                              const MeasurementT& measurement) {
  for (const auto& integrator_name : integrator_names) {
    if (auto* integrator = getIntegrator(integrator_name); integrator) {
      integrator->integrate(measurement);
    } else {
      return false;
    }
  }
  return true;
}

template <typename MeasurementT>
bool Pipeline::runPipeline(const std::vector<std::string>& integrator_names,
                           const MeasurementT& measurement) {
  const bool success = runIntegrators(integrator_names, measurement);
  runOperations();
  return success;
}

template <typename MeasurementA, typename MeasurementB, typename MeasurementC>
bool UnderwaterPipeline::runIntegrators(const std::vector<std::string>& integrator_names,
                              const MeasurementA& posed_depth,
                              const MeasurementB& depth_uncertainty,
                              const MeasurementC& pose_covariance) {
  for (const auto& integrator_name : integrator_names) {
    if (auto* integrator = getIntegrator(integrator_name); integrator) {
      integrator->integrate(posed_depth, depth_uncertainty, pose_covariance);
    } else {
      return false;
    }
  }
  return true;
}

template <typename MeasurementA, typename MeasurementB, typename MeasurementC>
bool UnderwaterPipeline::runPipeline(const std::vector<std::string>& integrator_names,
                           const MeasurementA& posed_depth,
                           const MeasurementB& depth_uncertainty,
                           const MeasurementC& pose_covariance) {
  const bool success = runIntegrators(integrator_names, posed_depth, depth_uncertainty, pose_covariance);
  runOperations();
  return success;
}
}  // namespace wavemap

#endif  // WAVEMAP_PIPELINE_IMPL_PIPELINE_INL_H_
