cmake_minimum_required(VERSION 3.15)
project(lob_sim_python)

# Set C++ Standard (C++20 for std::format)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
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

