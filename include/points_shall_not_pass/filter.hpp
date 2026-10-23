#pragma once

#include "points_shall_not_pass/utils/cuda_pointcloud_converter.hpp"
#include "points_shall_not_pass/utils/rigid_transform.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <cuda_runtime.h>
#include <thrust/memory.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>
#include <string>

class FilterNode : public rclcpp::Node
{
private:
    // ROS
    std::string input_topic;
    std::string filtered_topic;
    std::string frame_id;           // expected frame of the input cloud
    std::string output_frame_id;    // frame of the published cloud when the transform is applied

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub;

    // Passthrough limits; an axis is filtered only when down < up
    double downFilterLimitX, upFilterLimitX;
    double downFilterLimitY, upFilterLimitY;
    double downFilterLimitZ, upFilterLimitZ;

    // Static frame_id -> output_frame_id transform, applied before filtering
    RigidTransform input_to_output;
    bool transform_identity = true;

    // GPU
    std::unique_ptr<IConverter> converter_;
    cudaStream_t compute_stream = nullptr;

    thrust::device_vector<float> d_input;           // dense xyz
    thrust::device_vector<float> d_output;          // dense xyz, filtered
    thrust::device_vector<unsigned int> d_count;    // number of points in d_output

    void loadParameters();

    void ROSListenerInit();
    void ROSPublisherInit();

    void reserveAndResize(size_t totalElements);
    void standardPointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud);
    void runFilter(unsigned int inputSize, const std_msgs::msg::Header& header);
    void publishPc(const float* d_points, unsigned int size, const std_msgs::msg::Header& header);

public:
    FilterNode();
    ~FilterNode();

    void getInfo();
};
