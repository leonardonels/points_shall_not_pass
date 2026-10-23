#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/* p' = R p + t, R row-major. Static frame_id -> output_frame_id pose from config.yaml, no TF lookup. */
struct RigidTransform {
    float r[9];
    float t[3];
};

/* R = Rz(yaw) Ry(pitch) Rx(roll) (radians, the static_transform_publisher convention), then t. */
RigidTransform rigidFromXyzRpy(const double xyz[3], const double rpy[3]);

/* True if R = I and t = 0: the transform can be skipped. */
bool isIdentity(const RigidTransform& T);

/* Transforms n interleaved xyz points in place. Asynchronous on stream: no sync. */
cudaError_t transformPoints(const RigidTransform& T, std::uint32_t n, float* d_xyz, cudaStream_t stream);
