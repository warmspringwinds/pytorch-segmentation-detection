cmake_minimum_required(VERSION 2.6)


project (caffe2_cpp_tutorial)

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()


find_package(Protobuf REQUIRED)
find_package(CUDA QUIET)
find_library(GLOG_LIB glog)

find_library(CAFFE2_LIB caffe2 PATHS "/home/daniil/projects/speedup/caffe2_build/lib")
find_library(CAFFE2_GPU_LIB caffe2_gpu PATHS "/home/daniil/projects/speedup/caffe2_build/lib")
find_library(CUDNN_LIB cudnn PATHS "/home/daniil/cuda/lib64")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories("/home/daniil/projects/speedup/caffe2_build/include/")

# Adding cudnn include and library folders
include_directories("/home/daniil/cuda/include")


ADD_EXECUTABLE(benchmark benchmark.cpp)
TARGET_LINK_LIBRARIES(benchmark ${CAFFE2_LIB} ${CAFFE2_GPU_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${NCCL_LIB} ${CUDNN_LIB})

ADD_EXECUTABLE(read_allocated_gpu_memory read_allocated_gpu_memory.cpp)
TARGET_LINK_LIBRARIES(read_allocated_gpu_memory ${CAFFE2_LIB} ${CAFFE2_GPU_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${NCCL_LIB} ${CUDNN_LIB})

ADD_EXECUTABLE(inference inference.cpp)
TARGET_LINK_LIBRARIES(inference ${CAFFE2_LIB} ${CAFFE2_GPU_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${NCCL_LIB} ${CUDNN_LIB})