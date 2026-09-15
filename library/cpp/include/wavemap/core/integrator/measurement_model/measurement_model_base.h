#ifndef WAVEMAP_CORE_INTEGRATOR_MEASUREMENT_MODEL_MEASUREMENT_MODEL_BASE_H_
#define WAVEMAP_CORE_INTEGRATOR_MEASUREMENT_MODEL_MEASUREMENT_MODEL_BASE_H_

#include <memory>
#include <shared_mutex>
#include <mutex>
#include <map>
#include <unordered_map>

#include "wavemap/core/common.h"
#include "wavemap/core/config/type_selector.h"
#include "wavemap/core/data_structure/image.h"
#include "wavemap/core/integrator/projection_model/projector_base.h"
#include "wavemap/core/integrator/projective/update_type.h"

namespace wavemap {
struct MeasurementModelType : TypeSelector<MeasurementModelType> {
  using TypeSelector<MeasurementModelType>::TypeSelector;

  enum Id : TypeId { kConstantRay, kContinuousRay, kContinuousBeam, kContinuousLieRay };

  static constexpr std::array names = {"constant_ray", "continuous_ray",
                                       "continuous_beam", "continuous_lie_ray"};
};

struct BeamSelectorType : TypeSelector<BeamSelectorType> {
  using TypeSelector<BeamSelectorType>::TypeSelector;

  enum Id : TypeId { kNearestNeighbor, kAllNeighbors };

  static constexpr std::array names = {"nearest_neighbor", "all_neighbors"};
};

class MeasurementModelBase {
 public:
  using Ptr = std::shared_ptr<MeasurementModelBase>;
  using ConstPtr = std::shared_ptr<const MeasurementModelBase>;

  virtual ~MeasurementModelBase() = default;

  virtual FloatingPoint getPaddingAngle() const = 0;
  virtual FloatingPoint getPaddingSurfaceFront() const = 0;
  virtual FloatingPoint getPaddingSurfaceBack() const = 0;

  virtual FloatingPoint computeWorstCaseApproximationError(
      UpdateType update_type, FloatingPoint cell_to_sensor_distance,
      FloatingPoint cell_bounding_radius) const = 0;

  virtual FloatingPoint computeUpdate(
      const SensorCoordinates& sensor_coordinates) const = 0;
};

class MeasurementModelLieBase {
 public:
  using Ptr = std::shared_ptr<MeasurementModelLieBase>;
  using ConstPtr = std::shared_ptr<const MeasurementModelLieBase>;

  virtual ~MeasurementModelLieBase() = default;

  virtual FloatingPoint getPaddingAngle() const = 0;
  virtual FloatingPoint getPaddingSurfaceFront() const = 0;
  virtual FloatingPoint getPaddingSurfaceBack() const = 0;

  // internal map of Vector2D to const Eigen::Matrix<double, 6, 6> to hash the covariance matrices for each pixel in the depth image for a given image
  mutable std::map<std::pair<int, int>, Eigen::MatrixXd> covariance_map_;

  // internal map of Vector2D to const Eigen::Matrix<double, 6, 1> to hash the information matrices for each pixel in the depth image for a given image
  mutable std::map<std::pair<int, int>, Eigen::MatrixXd> information_map_;

  // internal shared mutex to protect the covariance and information maps from concurrent access
  mutable std::shared_mutex m_mutex;

  void clearInternalMaps() const {
    covariance_map_.clear();
    information_map_.clear();
  }

  // virtual void generateCovarianceAndInformationMatrices(
  //   const PosedImage<>::Ptr posed_depth_image,
  //   const Image<FloatingPoint>::Ptr depth_uncertainty_image,
  //   std::shared_ptr<MatrixT<double>> pose_covariance
  // ) const = 0;

  // virtual FloatingPoint computeWorstCaseApproximationError(
  //     UpdateType update_type, FloatingPoint cell_to_sensor_distance,
  //     FloatingPoint cell_bounding_radius) const = 0;

  virtual bool computeWorstCaseApproximationError(
      const SensorCoordinates& sensor_coordinates,
      const Point3D& r_pcenterz_c,
      const Point3D& r_pmaxz_c,
      const Point3D& r_pminz_c,
      const Eigen::Matrix4d& T_ac,
      UpdateType update_type,
      FloatingPoint bounding_sphere_radius) const = 0;

//   virtual FloatingPoint computeUpdate(
//       const SensorCoordinates& sensor_coordinates,
//       const Eigen::Matrix4d& T_ac) const = 0;

  virtual FloatingPoint computeUpdate(
      const SensorCoordinates& sensor_coordinates,
      const Point3D& r_pcenterz_c,
      const Point3D& r_pmaxz_c,
      const Eigen::Matrix4d& T_ac) const = 0;
};
}  // namespace wavemap

#endif  // WAVEMAP_CORE_INTEGRATOR_MEASUREMENT_MODEL_MEASUREMENT_MODEL_BASE_H_
