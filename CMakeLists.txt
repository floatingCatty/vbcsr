cmake_minimum_required(VERSION 3.15)
project(vbcsr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find MKL first, then BLAS
find_package(MKL)
if(MKL_FOUND)
    message(STATUS "MKL found: ${MKL_INCLUDE_DIRS}")
    include_directories(${MKL_INCLUDE_DIRS})
    link_directories(${MKL_LIBRARY_DIRS})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
else()
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
        add_definitions(-DUSE_BLAS)
    else()
        message(WARNING "No BLAS library found. Performance may be degraded.")
    endif()
endif()

# Pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from git...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Source files
set(SOURCES
    vbcsr/pybind_vbcsr.cpp
)

# Create the extension module
pybind11_add_module(vbcsr_core ${SOURCES})

# Link libraries
target_link_libraries(vbcsr_core PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

# Include directories
target_include_directories(vbcsr_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vbcsr/core)

# Install rule (optional, but good for pip install)
install(TARGETS vbcsr_core DESTINATION .)
