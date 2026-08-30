#include "octree_volume.h"

#include "check_macros.h"

#include <algorithm>
#include <cmath>

namespace {

// Safety cap on recursion depth: for leaf sizes >= 1 and volume dims of a
// few thousand voxels the natural depth is far below this.
constexpr int kMaxDepth = 24;

}  // namespace

OctreeVolume::~OctreeVolume()
{
    if (d_nodes_)
        CUDA_CHECK_NOEXCEPT(cudaFree(d_nodes_));
    if (d_child_)
        CUDA_CHECK_NOEXCEPT(cudaFree(d_child_));
}

void OctreeVolume::build(const Volume& volume, const TransferFunction& tf,
                         int leafSize, float scalarMin, float scalarMax, float epsilon)
{
    leafSize_ = std::max(1, leafSize);
    invRange_ = 1.0f / (scalarMax - scalarMin);

    nodes_.clear();
    children_.clear();
    numLeaves_ = numRelevant_ = numEmpty_ = 0;
    maxDepth_ = 0;

    buildNode(volume, tf, scalarMin, scalarMax, epsilon, leafSize_, 0,
              0, volume.dimX, 0, volume.dimY, 0, volume.dimZ);

    countStats();
    uploadToDevice();
}

// Recursively expands the node covering voxel range [x0,x1)x[y0,y1)x[z0,z1).
// Nodes are appended in preorder; internal nodes record a childOffset into
// the children_ array and append the 8 child node indices there, in octant
// order k = 0..7.
unsigned int OctreeVolume::buildNode(const Volume& volume, const TransferFunction& tf,
                                     float scalarMin, float scalarMax, float epsilon,
                                     int leafSize, int depth,
                                     int x0, int x1, int y0, int y1, int z0, int z1)
{
    maxDepth_ = std::max(maxDepth_, depth);

    const int sx = x1 - x0;
    const int sy = y1 - y0;
    const int sz = z1 - z0;

    // Classify the node by scanning its region of the dense grid, mapping
    // each voxel through the transfer function exactly like the renderer.
    float maxOp = 0.0f;
    unsigned long long total = 0;
    unsigned long long relevantCount = 0;
    if (sx > 0 && sy > 0 && sz > 0)
    {
        for (int vz = z0; vz < z1; ++vz)
            for (int vy = y0; vy < y1; ++vy)
                for (int vx = x0; vx < x1; ++vx)
                {
                    const float s = volume.data[
                        (static_cast<size_t>(vz) * volume.dimY + vy) * volume.dimX + vx];
                    const float tf_t = (s - scalarMin) * invRange_ * 2047.0f;
                    int idx = static_cast<int>(std::rint(tf_t));
                    idx = std::max(0, std::min(2047, idx));
                    const float op = tf.lut[idx].w;
                    if (op > maxOp) maxOp = op;
                    ++total;
                    if (op >= epsilon) ++relevantCount;
                }
    }

    unsigned int idx = static_cast<unsigned int>(nodes_.size());
    nodes_.emplace_back();
    OctreeNode& n = nodes_.back();
    n.relevant     = 0;
    n.isLeaf       = 1;
    n.childOffset  = 0;
    n.voxelMin[0]  = static_cast<unsigned int>(x0);
    n.voxelMin[1]  = static_cast<unsigned int>(y0);
    n.voxelMin[2]  = static_cast<unsigned int>(z0);
    n.voxelMax[0]  = static_cast<unsigned int>(x1);
    n.voxelMax[1]  = static_cast<unsigned int>(y1);
    n.voxelMax[2]  = static_cast<unsigned int>(z1);

    // Empty or degenerate (zero-volume child from an odd split) node.
    if (total == 0 || relevantCount == 0)
        return idx;

    // Stop subdividing once the node fits the leaf size, the depth cap is
    // reached, or the region is fully relevant (nothing to skip inside).
    const bool withinLeaf = sx <= leafSize && sy <= leafSize && sz <= leafSize;
    if (withinLeaf || depth >= kMaxDepth || relevantCount == total)
    {
        n.relevant = 1;
        return idx;
    }

    // Internal node: split into 8 octants.
    n.relevant = 1;
    n.isLeaf   = 0;
    n.childOffset = static_cast<unsigned int>(children_.size());
    const unsigned int childOff = n.childOffset;  // copy: `n` dangles after recursion
    const int midX = (x0 + x1) / 2;
    const int midY = (y0 + y1) / 2;
    const int midZ = (z0 + z1) / 2;

    // Reserve all 8 child slots up-front, before recursing into any child.
    // If the slots were reserved one-by-one interleaved with the recursion,
    // an internal child's own reservations could overlap the parent's
    // remaining (not-yet-written) slots at childOff + k.
    for (int k = 0; k < 8; ++k)
        children_.emplace_back(0);

    for (int k = 0; k < 8; ++k)
    {
        const int cx0 = (k & 1) ?            midX : x0;
        const int cx1 = (k & 1) ?            x1 : midX;
        const int cy0 = ((k >> 1) & 1) ?     midY : y0;
        const int cy1 = ((k >> 1) & 1) ?     y1 : midY;
        const int cz0 = ((k >> 2) & 1) ?     midZ : z0;
        const int cz1 = ((k >> 2) & 1) ?     z1 : midZ;
        const unsigned int childIdx =
            buildNode(volume, tf, scalarMin, scalarMax, epsilon, leafSize, depth + 1,
                      cx0, cx1, cy0, cy1, cz0, cz1);
        children_[childOff + k] = childIdx;
    }
    return idx;
}

void OctreeVolume::countStats()
{
    for (const OctreeNode& n : nodes_)
    {
        if (!n.isLeaf)
            continue;
        ++numLeaves_;
        if (n.relevant) ++numRelevant_;
        else            ++numEmpty_;
    }
}

void OctreeVolume::uploadToDevice()
{
    if (d_nodes_)
        return;

    const size_t nodeBytes = nodes_.size() * sizeof(OctreeNode);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nodes_), nodeBytes));
    CUDA_CHECK(cudaMemcpy(d_nodes_, nodes_.data(), nodeBytes, cudaMemcpyHostToDevice));

    if (!children_.empty())
    {
        const size_t childBytes = children_.size() * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_child_), childBytes));
        CUDA_CHECK(cudaMemcpy(d_child_, children_.data(), childBytes,
                              cudaMemcpyHostToDevice));
    }
}
