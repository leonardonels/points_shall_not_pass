#include "points_shall_not_pass/utils/rigid_transform.hpp"

#include <cmath>

namespace
{
    /* One thread per point: p' = R p + t. T is passed by value (48 bytes of kernel parameters). */
    __global__ void transformPointsKernel(RigidTransform T, std::uint32_t n, float* __restrict__ xyz)
    {
        const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        float* p = xyz + 3 * static_cast<std::size_t>(i);
        const float x = p[0], y = p[1], z = p[2];
        p[0] = T.r[0] * x + T.r[1] * y + T.r[2] * z + T.t[0];
        p[1] = T.r[3] * x + T.r[4] * y + T.r[5] * z + T.t[1];
        p[2] = T.r[6] * x + T.r[7] * y + T.r[8] * z + T.t[2];
    }
} // anonymous namespace

RigidTransform rigidFromXyzRpy(const double xyz[3], const double rpy[3])
{
    const double cr = std::cos(rpy[0]), sr = std::sin(rpy[0]);
    const double cp = std::cos(rpy[1]), sp = std::sin(rpy[1]);
    const double cy = std::cos(rpy[2]), sy = std::sin(rpy[2]);

    // Rz(yaw) * Ry(pitch) * Rx(roll), in double, rounded once.
    const double r[9] = {
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr,
    };
    RigidTransform T;
    for (int k = 0; k < 9; ++k) T.r[k] = static_cast<float>(r[k]);
    for (int k = 0; k < 3; ++k) T.t[k] = static_cast<float>(xyz[k]);
    return T;
}

bool isIdentity(const RigidTransform& T)
{
    for (int k = 0; k < 9; ++k)
        if (T.r[k] != (k % 4 == 0 ? 1.f : 0.f)) return false;
    return T.t[0] == 0.f && T.t[1] == 0.f && T.t[2] == 0.f;
}

cudaError_t transformPoints(const RigidTransform& T, std::uint32_t n, float* d_xyz, cudaStream_t stream)
{
    if (n == 0) return cudaSuccess;
    const unsigned threads = 256;
    const unsigned blocks = (n + threads - 1) / threads;
    transformPointsKernel<<<blocks, threads, 0, stream>>>(T, n, d_xyz);
    return cudaGetLastError();
}
