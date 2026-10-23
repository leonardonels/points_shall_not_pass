#pragma once

#include "points_shall_not_pass/utils/iconverter.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <cstddef>

class CudaConverter : public IConverter
{
private:
    thrust::device_vector<std::uint8_t> d_input_;
    cudaStream_t stream_ = nullptr;

public:
    CudaConverter();
    ~CudaConverter() override;

    CudaConverter(const CudaConverter&) = delete;
    CudaConverter& operator=(const CudaConverter&) = delete;

    unsigned int convert(const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud, thrust::device_vector<float>& d_out) override;
};
