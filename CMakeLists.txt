cmake_minimum_required(VERSION 3.28)
project(CUDA_TEST)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARIES}")  
message(STATUS "CUDA runtime libraries: ${CUDAToolkit_RUNTIME_LIBRARIES}")
message(STATUS "CUDA cublas libraries: ${CUDAToolkit_CUBLAS_LIBRARIES}")
message(STATUS "CUDA compiler: ${CUDAToolkit_NVCC_EXECUTABLE}")
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
add_executable(output output.cpp)
set_target_properties(output PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                         LINKER_LANGUAGE CUDA)
# set_target_properties(output PROPERTIES CUDA_ARCHITECTURES native
#                                         CUDA_SEPARABLE_COMPILATION ON
#                                         CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(output PRIVATE CUDA::cudart)                                    