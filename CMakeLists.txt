cmake_minimum_required(VERSION 3.10)

# Set your project name
project(Hivemind VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# compile server or client?
add_definitions(-DCLIENT=${CLIENT_VALUE})
message(STATUS "Compiling with CLIENT=${CLIENT_VALUE}")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")

if(TEST_NAME)
    list(APPEND SOURCES "tests/${TEST_NAME}.test.cpp")
    add_executable(${PROJECT_NAME} ${SOURCES})
else()
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/main.cpp")
    add_executable(Hivemind ${SOURCES})
endif()


# If you are using threads, find and link them
find_package(Threads REQUIRED)
target_link_libraries(Hivemind PRIVATE Threads::Threads)

