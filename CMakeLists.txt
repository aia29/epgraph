cmake_minimum_required(VERSION 3.12)

project(EPGraph)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS  "-O2 -DNDEBUG --std=c++17")

file(GLOB TEST_FILES "${CMAKE_SOURCE_DIR}/tests/*.cpp")
file(GLOB EXAMPLE_FILES "${CMAKE_SOURCE_DIR}/examples/*.cpp")

# Copy supplementary files for examples
execute_process(
    COMMAND bash -c "cp examples/diabetes.csv examples/plot_example_*.py ${CMAKE_CURRENT_BINARY_DIR}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Download Eigen
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0")
    message(WARNING "Downloading and extracting Eigen to ${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0")
    execute_process(
        COMMAND bash -c "mkdir -p third_party &&
                         cd third_party &&
                         wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&
                         tar -xf eigen-3.4.0.tar.gz"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

add_custom_target(executables)
foreach(FILE ${TEST_FILES} ${EXAMPLE_FILES})
    string(REPLACE ".cpp" "" TARGET_NAME "${FILE}")
    string(REPLACE "${CMAKE_SOURCE_DIR}/examples/" "" TARGET_NAME "${TARGET_NAME}")
    string(REPLACE "${CMAKE_SOURCE_DIR}/tests/" "" TARGET_NAME "${TARGET_NAME}")
    add_executable(${TARGET_NAME} ${FILE})
    target_include_directories(
        ${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0
    )
    add_dependencies(executables ${TARGET_NAME})
endforeach()
