cmake_minimum_required(VERSION 3.15)
project(lob_sim_python)

# Set C++ Standard (C++20 for std::format)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If PYTHON_EXECUTABLE is set (from setup.py), use it
if(DEFINED PYTHON_EXECUTABLE)
    message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
    set(PYBIND11_PYTHON_VERSION 3)
endif()

# Handle Python3_INCLUDE_DIR if provided
if(DEFINED Python3_INCLUDE_DIR AND EXISTS "${Python3_INCLUDE_DIR}")
    message(STATUS "Using provided Python3 include dir: ${Python3_INCLUDE_DIR}")
    set(Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIR})
endif()

# Find pybind11 first - it's better at Python detection
find_package(pybind11 CONFIG REQUIRED)

# For pybind11 2.x, we use pybind11_add_module which handles everything
# For pybind11 3.x, we need Python3 Development component
# Try to find it, but don't fail if we can't - pybind11 2.x doesn't need it
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(NOT Python3_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
endif()
if(NOT Python3_FOUND)
    find_package(Python3 COMPONENTS Interpreter QUIET)
endif()

message(STATUS "pybind11 version: ${pybind11_VERSION}")
if(Python3_FOUND)
    message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
endif()

# Create the Python Extension Module
pybind11_add_module(lob_sim bindings.cpp)

# Include directories
target_include_directories(lob_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# If we have Python include dirs, add them explicitly
if(Python3_INCLUDE_DIRS)
    target_include_directories(lob_sim PRIVATE ${Python3_INCLUDE_DIRS})
endif()

# Optimization Flags (Critical for HFT simulation performance)
if(MSVC)
    target_compile_options(lob_sim PRIVATE /O2 /Ob2)
else()
    target_compile_options(lob_sim PRIVATE -O3 -march=native -funroll-loops)
endif()

# Installation (optional - for pip install)
install(TARGETS lob_sim DESTINATION .)
