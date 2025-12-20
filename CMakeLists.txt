cmake_minimum_required(VERSION 3.10)
project(LOB_Sim)

set(CMAKE_CXX_STANDARD 17)

# Where are the header files?
include_directories(src)

# Create the executable
add_executable(lob_sim src/main.cpp src/OrderBook.cpp)