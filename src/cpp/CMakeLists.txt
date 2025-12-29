cmake_minimum_required(VERSION 3.15)
project(lob_sim_python)

# Set C++ Standard (C++20 for std::format)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
# If PYTHON_EXECUTABLE is set (from setup.py), use it to help find Python
if(DEFINED PYTHON_EXECUTABLE)
    message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
    # Set this for pybind11 to use
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

# Try to find Python3 to check for development headers
find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_FOUND AND NOT Python3_INCLUDE_DIRS)
    # Try to get include directory from Python itself
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
        OUTPUT_VARIABLE Python3_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(Python3_INCLUDE_DIR AND EXISTS "${Python3_INCLUDE_DIR}")
        set(Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIR})
        message(STATUS "Found Python3 include dir: ${Python3_INCLUDE_DIRS}")
    endif()
endif()

# pybind11 is better at finding Python, especially in virtual environments
# It will automatically use PYTHON_EXECUTABLE if set, or find Python itself
find_package(pybind11 CONFIG REQUIRED)

# If we still don't have Python include dirs, pybind11 should find them
if(NOT Python3_INCLUDE_DIRS)
    message(STATUS "Relying on pybind11 to find Python development headers...")
endif()

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

