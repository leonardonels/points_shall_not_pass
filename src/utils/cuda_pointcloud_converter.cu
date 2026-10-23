#include "points_shall_not_pass/utils/cuda_pointcloud_converter.hpp"

#include <string>

namespace
{

/* Packs x, y, z of every point into a dense XYZ float array. The byte offsets come
   from msg->fields, so any layout works: the map from pcd_to_pointcloud is x,y,z with
   point_step 12, a LiDAR driver is usually x,y,z,intensity,... with point_step 16+. */
__global__ void convertPointCloud2Kernel(
    const std::uint8_t* __restrict__ input_data,
    float*              __restrict__ output_data,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_step,
    std::uint32_t point_step,
    std::uint32_t x_offset,
    std::uint32_t y_offset,
    std::uint32_t z_offset)
{
    const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= width || r >= height) return;

    const std::uint8_t* src = input_data
        + static_cast<std::size_t>(r) * row_step
        + static_cast<std::size_t>(c) * point_step;

    float x, y, z;
    memcpy(&x, src + x_offset, sizeof(float));
    memcpy(&y, src + y_offset, sizeof(float));
    memcpy(&z, src + z_offset, sizeof(float));

    const std::uint64_t out_idx = (static_cast<std::uint64_t>(r) * width + c) * 3;

    output_data[out_idx + 0] = x;
    output_data[out_idx + 1] = y;
    output_data[out_idx + 2] = z;
}

/* Byte offset of a FLOAT32 field, or -1 if the cloud does not have it. */
long fieldOffset(const sensor_msgs::msg::PointCloud2& cloud, const std::string& name)
{
    for (const auto& f : cloud.fields)
        if (f.name == name && f.datatype == sensor_msgs::msg::PointField::FLOAT32)
            return static_cast<long>(f.offset);
    return -1;
}

} // anonymous namespace

CudaConverter::CudaConverter()
{
    cudaStreamCreate(&stream_);
}

CudaConverter::~CudaConverter()
{
    cudaStreamDestroy(stream_);
}

unsigned int CudaConverter::convert(
    const sensor_msgs::msg::PointCloud2::SharedPtr& sub_cloud,
    thrust::device_vector<float>& d_out)
{
    const std::uint32_t width = sub_cloud->width;
    const std::uint32_t height = sub_cloud->height;
    if (width == 0 || height == 0) return 0;

    const long x_offset = fieldOffset(*sub_cloud, "x");
    const long y_offset = fieldOffset(*sub_cloud, "y");
    const long z_offset = fieldOffset(*sub_cloud, "z");
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) return 0;   // no float x/y/z fields

    const std::size_t in_bytes = sub_cloud->data.size();
    const std::size_t out_floats = static_cast<std::size_t>(width) * height * 3;

    if(d_input_.size()< in_bytes)
        d_input_.resize(in_bytes);  // filled with zeros

    if (d_out.size() < out_floats)
        d_out.resize(out_floats);   // filled with zeros

    cudaMemcpyAsync(thrust::raw_pointer_cast(d_input_.data()), sub_cloud->data.data(), in_bytes, cudaMemcpyHostToDevice, stream_);

    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);

    convertPointCloud2Kernel<<<blocks, threads, 0, stream_>>>(
        thrust::raw_pointer_cast(d_input_.data()),
        thrust::raw_pointer_cast(d_out.data()),
        width, height,
        sub_cloud->row_step, sub_cloud->point_step,
        static_cast<std::uint32_t>(x_offset),
        static_cast<std::uint32_t>(y_offset),
        static_cast<std::uint32_t>(z_offset));

    cudaStreamSynchronize(stream_);
    return static_cast<unsigned int>(width * height);
}
