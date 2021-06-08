cmake_minimum_required(VERSION 3.14)

project(D3D12Base VERSION 0.1)

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

add_subdirectory(D3D12BaseLib)
add_subdirectory(sample_projects)