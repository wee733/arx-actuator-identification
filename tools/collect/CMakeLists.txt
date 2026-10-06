cmake_minimum_required(VERSION 3.16)
project(arx_collect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Root of the ARX R5 SDK controller package (contains lib/arx_r5_src and
# lib/arx_hardware_interface), e.g. .../ARX_R5_ros2/arx_r5_controller.
set(ARX_SDK_ROOT "" CACHE PATH "ARX R5 SDK controller directory")
if(NOT EXISTS "${ARX_SDK_ROOT}/lib/arx_r5_src/include")
    message(FATAL_ERROR "Set -DARX_SDK_ROOT to the ARX R5 SDK controller directory")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARX_LIB_DEFAULT "${ARX_SDK_ROOT}/lib/arx_r5_src/libarx_r5_src-arm64.so")
else()
    set(ARX_LIB_DEFAULT "${ARX_SDK_ROOT}/lib/arx_r5_src/libarx_r5_src.so")
endif()
set(ARX_LIB "${ARX_LIB_DEFAULT}" CACHE FILEPATH "ARX SDK shared library")
get_filename_component(ARX_LIB_DIR "${ARX_LIB}" DIRECTORY)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(arx_collect arx_collect.cpp)
target_include_directories(arx_collect PRIVATE
    "${ARX_SDK_ROOT}/lib/arx_r5_src/include"
    "${ARX_SDK_ROOT}/lib/arx_hardware_interface/include")
target_link_libraries(arx_collect PRIVATE
    Boost::program_options Threads::Threads Eigen3::Eigen "${ARX_LIB}")

# The SDK library resolves ROS runtime libraries (kdl_parser, urdf, ...) through
# its own dependencies; keep their directories on the runtime path so the
# collector runs without sourcing a ROS environment first.
set(RUNTIME_DIRS "${ARX_LIB_DIR}")
foreach(lib kdl_parser urdf rcutils ament_index_cpp class_loader rcpputils)
    find_library(_dep_${lib} NAMES ${lib} PATHS /opt/ros/humble /opt/ros/jazzy PATH_SUFFIXES lib)
    if(_dep_${lib})
        get_filename_component(_dir "${_dep_${lib}}" DIRECTORY)
        list(APPEND RUNTIME_DIRS "${_dir}")
    endif()
endforeach()
list(REMOVE_DUPLICATES RUNTIME_DIRS)
set_target_properties(arx_collect PROPERTIES BUILD_RPATH "${RUNTIME_DIRS}" INSTALL_RPATH "${RUNTIME_DIRS}")
target_link_options(arx_collect PRIVATE -Wl,--disable-new-dtags)
