#ifndef WAVEMAP_CORE_MAP_IMPL_HASHED_CHUNKED_WAVELET_OCTREE_INL_H_
#define WAVEMAP_CORE_MAP_IMPL_HASHED_CHUNKED_WAVELET_OCTREE_INL_H_

#include "wavemap/core/indexing/index_conversions.h"
#include <wavemap/core/utils/iterate/grid_iterator.h>

namespace wavemap {
inline size_t HashedChunkedWaveletOctree::size() const {
  size_t size = 0u;
  forEachBlock([&size](const BlockIndex& /*block_index*/, const Block& block) {
    size += block.size();
  });
  return size;
}

inline FloatingPoint HashedChunkedWaveletOctree::getCellValue(
    const Index3D& index) const {
  const BlockIndex block_index =
      convert::indexToBlockIndex(index, config_.tree_height);
  const Block* block = getBlock(block_index);
  if (!block) {
    return 0.f;
  }
  const CellIndex cell_index = indexToCellIndex({0, index});
  return block->getCellValue(cell_index);
}

inline FloatingPoint HashedChunkedWaveletOctree::getCellValue(
    const OctreeIndex& index) const {
  const BlockIndex block_index = indexToBlockIndex(index);
  const Block* block = getBlock(block_index);
  if (!block) {
    return 0.f;
  }
  const CellIndex cell_index = indexToCellIndex(index);
  return block->getCellValue(cell_index);
}

inline void HashedChunkedWaveletOctree::setCellValue(const Index3D& index,
                                                     FloatingPoint new_value) {
  const BlockIndex block_index =
      convert::indexToBlockIndex(index, config_.tree_height);
  auto& block = getOrAllocateBlock(block_index);
  const CellIndex cell_index = indexToCellIndex({0, index});
  block.setCellValue(cell_index, new_value);
}

inline void HashedChunkedWaveletOctree::addToCellValue(const Index3D& index,
                                                       FloatingPoint update) {
  const BlockIndex block_index =
      convert::indexToBlockIndex(index, config_.tree_height);
  auto& block = getOrAllocateBlock(block_index);
  const CellIndex cell_index = indexToCellIndex({0, index});
  block.addToCellValue(cell_index, update);
}

inline bool HashedChunkedWaveletOctree::hasBlock(
    const Index3D& block_index) const {
  return block_map_.hasBlock(block_index);
}

inline bool HashedChunkedWaveletOctree::eraseBlock(
    const HashedChunkedWaveletOctree::BlockIndex& block_index) {
  return block_map_.eraseBlock(block_index);
}

template <typename IndexedBlockVisitor>
void HashedChunkedWaveletOctree::eraseBlockIf(
    IndexedBlockVisitor indicator_fn) {
  block_map_.eraseBlockIf(indicator_fn);
}

inline HashedChunkedWaveletOctree::Block* HashedChunkedWaveletOctree::getBlock(
    const Index3D& block_index) {
  return block_map_.getBlock(block_index);
}

inline const HashedChunkedWaveletOctree::Block*
HashedChunkedWaveletOctree::getBlock(const Index3D& block_index) const {
  return block_map_.getBlock(block_index);
}

inline HashedChunkedWaveletOctree::Block&
HashedChunkedWaveletOctree::getOrAllocateBlock(const Index3D& block_index) {
  return block_map_.getOrAllocateBlock(block_index, config_.tree_height,
                                       config_.min_log_odds,
                                       config_.max_log_odds);
}

template <typename IndexedBlockVisitor>
void HashedChunkedWaveletOctree::forEachBlock(IndexedBlockVisitor visitor_fn) {
  block_map_.forEachBlock(visitor_fn);
}

template <typename IndexedBlockVisitor>
void HashedChunkedWaveletOctree::forEachBlock(
    IndexedBlockVisitor visitor_fn) const {
  block_map_.forEachBlock(visitor_fn);
}

inline HashedChunkedWaveletOctree::BlockIndex
HashedChunkedWaveletOctree::indexToBlockIndex(
    const OctreeIndex& node_index) const {
  const Index3D index = convert::nodeIndexToMinCornerIndex(node_index);
  return convert::indexToBlockIndex(index, config_.tree_height);
}

inline HashedChunkedWaveletOctree::CellIndex
HashedChunkedWaveletOctree::indexToCellIndex(OctreeIndex index) const {
  DCHECK_LE(index.height, config_.tree_height);
  const IndexElement height_difference = config_.tree_height - index.height;
  index.position =
      int_math::div_exp2_floor_remainder(index.position, height_difference);
  return index;
}

inline Eigen::Matrix3Xf HashedChunkedWaveletOctree::
    getOccupiedCellsPointcloud(FloatingPoint occupancy_threshold_log_odds) const {
  std::vector<Point3D> pointcloud_vec;

  // If the map is empty, there's no work to do
  if (this->empty()) {
    return Eigen::Matrix3Xf(3, 0);
  }

  // Define a functor that converts map leaf nodes into pointcloud_vec points
  auto add_points_for_leaf_node =
      [min_cell_width = this->getMinCellWidth(),
       occupancy_threshold = occupancy_threshold_log_odds,
       &pointcloud_vec](
          const OctreeIndex& node_index, FloatingPoint node_log_odds) {
        if (occupancy_threshold < node_log_odds) {
          if (node_index.height == 0) {
            const Point3D center = convert::indexToCenterPoint(
                node_index.position, min_cell_width);
            pointcloud_vec.emplace_back(center);
          } else {
            const Index3D node_min_corner =
                convert::nodeIndexToMinCornerIndex(node_index);
            const Index3D node_max_corner =
                convert::nodeIndexToMaxCornerIndex(node_index);
            for (const Index3D& index :
                 Grid(node_min_corner, node_max_corner)) {
              const Point3D center =
                  convert::indexToCenterPoint(index, min_cell_width);
              pointcloud_vec.emplace_back(center);
            }
          }
        }
      };

  // Define a functor that allows us to skip unchanged map blocks
  auto process_block_if_changed = [this, &add_points_for_leaf_node](
                                      const Index3D& block_index,
                                      const auto& block) {    
    // here, we want to process all blocks regardless of whether they have changed
    block.forEachLeaf(block_index, add_points_for_leaf_node);
  };

  // Iterate over the map
  // if (const auto* hashed_wavelet_octree =
  //         dynamic_cast<const HashedWaveletOctree*>(occupancy_map_.get());
  //     hashed_wavelet_octree) {
  //   hashed_wavelet_octree->forEachBlock(process_block_if_changed);
  // } else if (const auto* hashed_chunked_wavelet_octree =
  //                dynamic_cast<const HashedChunkedWaveletOctree*>(
  //                    occupancy_map_.get());
  //            hashed_chunked_wavelet_octree) {
  //   hashed_chunked_wavelet_octree->forEachBlock(process_block_if_changed);
  // } else {  // Fallback for non-hashed map types: simply process all leaves
  //   occupancy_map_->forEachLeaf(add_points_for_leaf_node);
  // }
  this->forEachBlock(process_block_if_changed);

  // convert vector of 3D points to a Eigen::Matrix3Xd pointcloud
  Eigen::Matrix3Xf pointcloud(3, pointcloud_vec.size());
  for (size_t i = 0; i < pointcloud_vec.size(); ++i) {
    pointcloud.col(i) = pointcloud_vec[i];
  }

  // Publish the msg
  return pointcloud;
}
}  // namespace wavemap

#endif  // WAVEMAP_CORE_MAP_IMPL_HASHED_CHUNKED_WAVELET_OCTREE_INL_H_
