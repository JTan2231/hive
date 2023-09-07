cmake_minimum_required(VERSION 3.10)

# Set your project name
project(Hivemind VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add executable
file(GLOB SOURCES "src/*.cpp")
list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/main.cpp")
add_executable(Hivemind ${SOURCES})

# If you are using threads, find and link them
find_package(Threads REQUIRED)
target_link_libraries(Hivemind PRIVATE Threads::Threads)

