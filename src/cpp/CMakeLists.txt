cmake_minimum_required(VERSION 3.15)
project(lob_sim_python)

# Set C++ Standard (C++20 for std::format)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRITICAL: For pybind11 3.0+, we MUST find Python3 with Development component
# BEFORE loading pybind11. This provides python3_add_library command.

# If PYTHON_EXECUTABLE is set (from setup.py), use it
if(DEFINED PYTHON_EXECUTABLE)
    message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

# Set hints from environment variables if provided
if(DEFINED Python3_INCLUDE_DIR)
    set(Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIR})
endif()

# Find Python3 with Development component - REQUIRED for pybind11 3.0+
# Try Development.Module first (works without full Python library)
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)

if(NOT Python3_FOUND)
    # Fall back to full Development (older CMake or missing Development.Module)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
endif()

if(NOT Python3_FOUND)
    # Last resort: just Interpreter, let pybind11 handle the rest
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    message(WARNING "Python3 Development component not found. Build may fail.")
endif()

message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

# Now find pybind11 - it will use the Python3 we just found
find_package(pybind11 CONFIG REQUIRED)

# Create the Python Extension Module
pybind11_add_module(lob_sim bindings.cpp)

# Include directories
target_include_directories(lob_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optimization Flags (Critical for HFT simulation performance)
if(MSVC)
    target_compile_options(lob_sim PRIVATE /O2 /Ob2)
else()
    target_compile_options(lob_sim PRIVATE -O3 -march=native -funroll-loops)
endif()

# Installation (optional - for pip install)
install(TARGETS lob_sim DESTINATION .)

