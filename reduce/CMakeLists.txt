cmake_minimum_required(VERSION 3.20)
# set project name is directory name
get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${ProjectId} VERSION 1.0.0 LANGUAGES CXX CUDA) # Enable CUDA as a language
find_package(CUDAToolkit REQUIRED)
# set executable name is file name
add_executable(${ProjectId} ${ProjectId}.cu) # Change the file extension to .cu
target_link_libraries(${ProjectId} PRIVATE CUDA::cudart)
message("Project id: ${ProjectId}")
set_target_properties(${ProjectId} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)