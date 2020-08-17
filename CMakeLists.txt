cmake_minimum_required(VERSION 3.0)

project(archit)

find_package(CUDA 10.2 REQUIRED)

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(LIBTORCH "lib/libtorch")

# Below will append libtorch to path so CMake can see files
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH}")

# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT REQUIRED 7.0)
find_package(Torch REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#if(TORCH_FOUND)
#  message(STATUS "Torch IncludeDir ${Torch_INCLUDE_DIRS}")
#endif (TORCH_FOUND)
if (TENSORRT_FOUND)
  message(STATUS "TensorRT IncludeDir ${TensorRT_INCLUDE_DIRS}")
  message(STATUS "TensorRT Version ${TensorRT_VERSION_STRING}")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT FOUND\n")
  set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS}
)

add_executable(archit_inf src/main.cpp
			  include/cifar10.h
			  src/cifar10.cpp)
		  		
message(STATUS "TensorRT Library is ${TensorRT_LIBRARY}")

target_link_libraries(archit_inf
    ${TensorRT_LIBRARY}
    ${CUDA_LIBRARIES}
    "/usr/lib/x86_64-linux-gnu/libnvparsers.so"
    "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so"
)
target_link_libraries(archit_inf ${TORCH_LIBRARIES} )
#set_property(TARGET archit-inf  PROPERTY CXX_STANDARD 14)
