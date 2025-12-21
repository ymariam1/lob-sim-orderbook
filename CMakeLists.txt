cmake_minimum_required(VERSION 3.10)
project(LOB_Sim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

add_executable(lob_sim src/main.cpp src/OrderBook.cpp)