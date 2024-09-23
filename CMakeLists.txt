cmake_minimum_required(VERSION 3.10)
project(HomeworkWeek6 CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the path to libtorch
set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)

# Set CUDA compiler path
set(CMAKE_CUDA_COMPILER $ENV{HOME}/miniconda3/envs/capstone/bin/nvcc)

# Set NVTOOLSEXT library path
# find $HOME/miniconda3/envs/capstone -n "*libnvToolsExt.so" 2>/dev/null
set(LIBNVTOOLSEXT $ENV{HOME}/miniconda3/envs/capstone/lib/libnvToolsExt.so)

# Find Torch package
find_package(Torch REQUIRED)
#find_package(CUDA REQUIRED)

# Add your executable
add_executable(main ./src/neural_network.cpp ./src/basic_tensor.cpp ./src/main.cpp )

# Include directories
target_include_directories(main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
# message(${TORCH_INCLUDE_DIRS})
# message(${CUDA_INCLUDE_DIRS})
# Link against Torch and CUDA libraries
target_link_libraries(main 
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)
#message(${TORCH_LIBRARIES})
# message(${CUDA_LIBRARIES})
#${LIBNVTOOLSEXT}

# Add Torch compile definitions
#target_compile_definitions(main PRIVATE ${TORCH_DEFINITIONS})

# Set CUDA architectures
#set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 86)

# Set RPATH for Linux
# if(UNIX AND NOT APPLE)
#     set_target_properties(main PROPERTIES
#         INSTALL_RPATH_USE_LINK_PATH TRUE
#         BUILD_WITH_INSTALL_RPATH TRUE
#     )
# endif()