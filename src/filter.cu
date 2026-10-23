#include "points_shall_not_pass/filter.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include <thrust/device_ptr.h>

namespace
{
    /* One thread per point: keeps the points inside the enabled [down, up] ranges.
       The output is compacted through an atomic counter, so the point order is not preserved. */
    __global__ void passthroughFilterKernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        unsigned int* __restrict__ d_count,
        unsigned int n,
        float downX, float upX, bool doX,
        float downY, float upY, bool doY,
        float downZ, float upZ, bool doZ)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        float x = input[i * 3 + 0];
        float y = input[i * 3 + 1];
        float z = input[i * 3 + 2];

        if (doX && (x < downX || x > upX)) return;
        if (doY && (y < downY || y > upY)) return;
        if (doZ && (z < downZ || z > upZ)) return;

        unsigned int write_idx = atomicAdd(d_count, 1);
        output[write_idx * 3 + 0] = x;
        output[write_idx * 3 + 1] = y;
        output[write_idx * 3 + 2] = z;
    }
} // anonymous namespace

FilterNode::FilterNode() : Node("gandalf_node")
{
    this->getInfo();
    this->loadParameters();

    this->converter_ = std::make_unique<CudaConverter>();
    this->d_count.resize(1);

    cudaStreamCreate(&compute_stream);

    this->ROSPublisherInit();
    this->ROSListenerInit();
}

FilterNode::~FilterNode()
{
    if (compute_stream) cudaStreamDestroy(compute_stream);
}

void FilterNode::getInfo()
{
    cudaDeviceProp prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    RCLCPP_INFO(this->get_logger(), "GPU has cuda devices: %d", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        RCLCPP_INFO(this->get_logger(), "----device id: %d info----", i);
        RCLCPP_INFO(this->get_logger(), "  GPU : %s", prop.name);
        RCLCPP_INFO(this->get_logger(), "  Capability: %d.%d", prop.major, prop.minor);
        RCLCPP_INFO(this->get_logger(), "  Avaiable Streaming Multiprocessors: %d", prop.multiProcessorCount);
        RCLCPP_INFO(this->get_logger(), "  Global memory: %luMB", prop.totalGlobalMem >> 20);
        RCLCPP_INFO(this->get_logger(), "  Shared memory in a block: %luKB", prop.sharedMemPerBlock >> 10);
        RCLCPP_INFO(this->get_logger(), "  warp size: %d", prop.warpSize);
        RCLCPP_INFO(this->get_logger(), "  threads in a block: %d", prop.maxThreadsPerBlock);
        RCLCPP_INFO(this->get_logger(), "  block dim: (%d,%d,%d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        RCLCPP_INFO(this->get_logger(), "  grid dim: (%d,%d,%d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        RCLCPP_INFO(this->get_logger(), "-------------------------");
    }
}

void FilterNode::loadParameters()
{
    declare_parameter("input_topic", "/pointcloud");
    declare_parameter("filtered_topic", "/filtered_points");
    declare_parameter("frame_id", "lidar_link");
    declare_parameter("output_frame_id", "base_link");
    declare_parameter("input_to_output_xyz", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter("input_to_output_rpy", std::vector<double>{0.0, 0.0, 0.0});

    declare_parameter("downFilterLimitX", -1e10);
    declare_parameter("upFilterLimitX", 1e10);
    declare_parameter("downFilterLimitY", -1e10);
    declare_parameter("upFilterLimitY", 1e10);
    declare_parameter("downFilterLimitZ", 0.0);
    declare_parameter("upFilterLimitZ", 0.0);

    get_parameter("input_topic", this->input_topic);
    get_parameter("filtered_topic", this->filtered_topic);
    get_parameter("frame_id", this->frame_id);
    get_parameter("output_frame_id", this->output_frame_id);

    get_parameter("downFilterLimitX", this->downFilterLimitX);
    get_parameter("upFilterLimitX", this->upFilterLimitX);
    get_parameter("downFilterLimitY", this->downFilterLimitY);
    get_parameter("upFilterLimitY", this->upFilterLimitY);
    get_parameter("downFilterLimitZ", this->downFilterLimitZ);
    get_parameter("upFilterLimitZ", this->upFilterLimitZ);

    {
        std::vector<double> xyz, rpy;
        get_parameter("input_to_output_xyz", xyz);
        get_parameter("input_to_output_rpy", rpy);
        if (xyz.size() != 3 || rpy.size() != 3) {
            RCLCPP_ERROR(this->get_logger(), "input_to_output_xyz and input_to_output_rpy need 3 values each, using identity");
            xyz.assign(3, 0.0);
            rpy.assign(3, 0.0);
        }
        this->input_to_output = rigidFromXyzRpy(xyz.data(), rpy.data());
        this->transform_identity = isIdentity(this->input_to_output);
        RCLCPP_INFO(this->get_logger(), "Transform %s -> %s%s", this->frame_id.c_str(), this->output_frame_id.c_str(),
                    this->transform_identity ? " (identity, skipped)" : "");
    }

    RCLCPP_INFO(this->get_logger(), "Passthrough X [%g, %g]%s", downFilterLimitX, upFilterLimitX, downFilterLimitX < upFilterLimitX ? "" : " (off)");
    RCLCPP_INFO(this->get_logger(), "Passthrough Y [%g, %g]%s", downFilterLimitY, upFilterLimitY, downFilterLimitY < upFilterLimitY ? "" : " (off)");
    RCLCPP_INFO(this->get_logger(), "Passthrough Z [%g, %g]%s", downFilterLimitZ, upFilterLimitZ, downFilterLimitZ < upFilterLimitZ ? "" : " (off)");
}

void FilterNode::ROSPublisherInit()
{
    this->pcl_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(this->filtered_topic, rclcpp::QoS(1).reliable());
    RCLCPP_INFO(this->get_logger(), "Publishing pcl filtered on: %s", this->filtered_topic.c_str());
}

void FilterNode::ROSListenerInit()
{
    this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        this->input_topic, rclcpp::SensorDataQoS(),
        std::bind(&FilterNode::standardPointcloud, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to topic: %s", this->input_topic.c_str());
}

void FilterNode::reserveAndResize(size_t totalElements)
{
    // removes the malloc logic inside .resize() by reserving enough memory beforehand
    if (d_input.capacity() < totalElements) {
        d_input.reserve(totalElements);
        d_output.reserve(totalElements);
    }

    d_input.resize(totalElements);
    d_output.resize(totalElements);
}

void FilterNode::standardPointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr sub_cloud)
{
    unsigned int inputSize = sub_cloud->width * sub_cloud->height;
    size_t totalElements = static_cast<size_t>(inputSize) * 3;     // XYZ

    reserveAndResize(totalElements);

    // The converter synchronizes its own stream: d_input is ready for compute_stream.
    inputSize = converter_->convert(sub_cloud, d_input);
    if (inputSize == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Empty cloud or no FLOAT32 x/y/z fields, skipping");
        return;
    }

    std_msgs::msg::Header header = sub_cloud->header;

    // Into the output frame before filtering: the limits are expressed in output_frame_id.
    if (!transform_identity) {
        if (header.frame_id != frame_id) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Cloud frame '%s' differs from frame_id '%s', applying the transform anyway",
                header.frame_id.c_str(), frame_id.c_str());
        }
        const cudaError_t err = transformPoints(input_to_output, inputSize, thrust::raw_pointer_cast(d_input.data()),
                                                compute_stream);
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "%s -> %s transform failed: %s",
                         frame_id.c_str(), output_frame_id.c_str(), cudaGetErrorString(err));
            return;
        }
        header.frame_id = output_frame_id;
    }

    runFilter(inputSize, header);
}

void FilterNode::runFilter(unsigned int inputSize, const std_msgs::msg::Header& header)
{
    if (inputSize == 0) return;

    const unsigned int threads = 256;
    const unsigned int blocks = (inputSize + threads - 1) / threads;

    unsigned int* d_count_ptr = thrust::raw_pointer_cast(d_count.data());
    cudaMemsetAsync(d_count_ptr, 0, sizeof(unsigned int), compute_stream);

    passthroughFilterKernel<<<blocks, threads, 0, compute_stream>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data()),
        d_count_ptr,
        inputSize,
        static_cast<float>(downFilterLimitX), static_cast<float>(upFilterLimitX), downFilterLimitX < upFilterLimitX,
        static_cast<float>(downFilterLimitY), static_cast<float>(upFilterLimitY), downFilterLimitY < upFilterLimitY,
        static_cast<float>(downFilterLimitZ), static_cast<float>(upFilterLimitZ), downFilterLimitZ < upFilterLimitZ);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        RCLCPP_ERROR(this->get_logger(), "Passthrough filter launch failed: %s", cudaGetErrorString(err));
        return;
    }

    unsigned int outputSize = 0;
    cudaMemcpyAsync(&outputSize, d_count_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);

    this->publishPc(thrust::raw_pointer_cast(d_output.data()), outputSize, header);
}

void FilterNode::publishPc(const float* d_points, unsigned int size, const std_msgs::msg::Header& header)
{
    sensor_msgs::msg::PointCloud2 pc;
    pc.header = header;
    pc.height = 1;
    pc.width = size;
    pc.is_bigendian = false;
    pc.is_dense = true;
    pc.point_step = 3 * sizeof(float);
    pc.row_step = pc.point_step * size;

    const char* names[3] = {"x", "y", "z"};
    pc.fields.resize(3);
    for (int k = 0; k < 3; ++k) {
        pc.fields[k].name = names[k];
        pc.fields[k].offset = k * sizeof(float);
        pc.fields[k].datatype = sensor_msgs::msg::PointField::FLOAT32;
        pc.fields[k].count = 1;
    }

    // Straight from the device into the message buffer: same dense xyz layout.
    pc.data.resize(pc.row_step);
    if (size > 0) {
        cudaMemcpyAsync(pc.data.data(), d_points, pc.row_step, cudaMemcpyDeviceToHost, compute_stream);
        cudaStreamSynchronize(compute_stream);
    }

    pcl_pub->publish(pc);
}
