#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cstddef>

class IConverter
{
public:
    virtual unsigned int convert(
        const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
        thrust::device_vector<float>& d_out) = 0;

    virtual ~IConverter() = default;
};
