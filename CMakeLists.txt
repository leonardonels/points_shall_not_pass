cmake_minimum_required(VERSION 3.18)
project(points_shall_not_pass LANGUAGES CXX CUDA)

# Host-only warnings: nvcc does not understand them.
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(POINTS_SHALL_NOT_PASS_CUDA_ARCH "89" CACHE STRING "CUDA arch for the GPU backend (RTX 6000 Ada = 89, Orin = 87)")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${POINTS_SHALL_NOT_PASS_CUDA_ARCH})
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA 13+ relocates CCCL (thrust/cub) under include/cccl. nvcc adds this path
# automatically, but host (.cpp) translation units that include thrust need it.
foreach(_cuda_inc ${CUDAToolkit_INCLUDE_DIRS})
  if(EXISTS "${_cuda_inc}/cccl")
    include_directories("${_cuda_inc}/cccl")
  endif()
endforeach()

# Pointcloud converter (CUDA)
add_library(pointcloud_converter src/utils/cuda_pointcloud_converter.cu)
target_compile_options(pointcloud_converter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
target_include_directories(pointcloud_converter PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(pointcloud_converter sensor_msgs)
target_link_libraries(pointcloud_converter CUDA::cudart)

# Static input -> output frame transform (CUDA)
add_library(rigid_transform src/utils/rigid_transform.cu)
target_compile_options(rigid_transform PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
target_include_directories(rigid_transform PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(rigid_transform CUDA::cudart)

# Node
add_library(${PROJECT_NAME} src/filter.cu)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs std_msgs)
target_link_libraries(${PROJECT_NAME}
  pointcloud_converter
  rigid_transform
  CUDA::cudart)

add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
