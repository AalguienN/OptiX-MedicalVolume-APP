#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <cuda_runtime.h>

#include <vector>

// Host-side construction of the SVO / Octree volume representation
// (Variant A, --mode octree). The octree is built once, offline, over the
// dense grid and acts purely as a spatial acceleration structure: each node
// records only whether its subtree contains any rendering-relevant voxel, and
// leaf nodes reference the half-open voxel region [voxelMin, voxelMax) of the
// unmodified dense voxel grid (which is not duplicated). Nodes are stored in
// a preorder array whose root sits at index 0; a separate array holds the 8
// child node indices of every internal node, so children are looked up by
// index rather than by contiguous layout.
class OctreeVolume
{
public:
    OctreeVolume() = default;
    ~OctreeVolume();

    OctreeVolume(const OctreeVolume&)            = delete;
    OctreeVolume& operator=(const OctreeVolume&) = delete;

    void build(const Volume& volume, const TransferFunction& tf,
               int leafSize, float scalarMin, float scalarMax, float epsilon);

    OctreeNode*   deviceNodes() const { return d_nodes_; }
    unsigned int* deviceChild() const { return d_child_; }

    const std::vector<OctreeNode>&  hostNodes()   const { return nodes_; }
    const std::vector<unsigned int>& hostChildren() const { return children_; }

    int          leafSize()    const { return leafSize_; }
    int          maxDepth()    const { return maxDepth_; }
    size_t       numNodes()    const { return nodes_.size(); }
    unsigned int numLeaves()   const { return numLeaves_; }
    unsigned int numRelevant() const { return numRelevant_; }
    unsigned int numEmpty()    const { return numEmpty_; }
    size_t       nodeBytes()   const { return nodes_.size() * sizeof(OctreeNode) +
                                              children_.size() * sizeof(unsigned int); }

private:
    unsigned int buildNode(const Volume& volume, const TransferFunction& tf,
                           float scalarMin, float scalarMax, float epsilon,
                           int leafSize, int depth,
                           int x0, int x1, int y0, int y1, int z0, int z1);
    void countStats();
    void uploadToDevice();

    std::vector<OctreeNode>  nodes_;
    std::vector<unsigned int> children_;

    OctreeNode*   d_nodes_ = nullptr;
    unsigned int* d_child_ = nullptr;

    int    leafSize_   = 0;
    int    maxDepth_   = 0;
    unsigned int numLeaves_  = 0;
    unsigned int numRelevant_ = 0;
    unsigned int numEmpty_   = 0;
    float  invRange_  = 0.0f;
};
