cmake_minimum_required(VERSION 3.20)

#----------------------------------------------------------------------------------------------------------------------------------

set(CUDASYS $ENV{CUDASYS})
set(CMAKE_CUDA_TOOLKIT_ROOT_DIR ${CUDASYS})
set(CMAKE_CUDA_COMPILER "${CUDASYS}/bin/nvcc")
link_directories("${CUDASYS}/lib64")
find_package(CUDAToolkit 13.2 REQUIRED COMPONENTS cudart curand cublas cusolver)

#----------------------------------------------------------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "gcc-15")
set(CMAKE_CXX_COMPILER "g++-15")
set(CMAKE_CUDA_HOST_COMPILER "g++-15")

#----------------------------------------------------------------------------------------------------------------------------------

project(QTensorNet LANGUAGES CXX CUDA)

#----------------------------------------------------------------------------------------------------------------------------------

find_package(MPI REQUIRED)

#----------------------------------------------------------------------------------------------------------------------------------

set(PETSC_DIR "$ENV{PETSC_DIR}")
set(PETSC_ARCH "$ENV{PETSC_ARCH}")
set(SLEPC_DIR "$ENV{SLEPC_DIR}")

if(NOT PETSC_DIR OR NOT PETSC_ARCH OR NOT SLEPC_DIR)
    message(FATAL_ERROR "PETSC_DIR, PETSC_ARCH, and SLEPC_DIR must be set in the environment")
endif()

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:${SLEPC_DIR}/${PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
pkg_check_modules(SLEPC REQUIRED IMPORTED_TARGET SLEPc)

#----------------------------------------------------------------------------------------------------------------------------------

find_package(yaml-cpp REQUIRED)

#----------------------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -fPIC -ffast-math -mavx")

#----------------------------------------------------------------------------------------------------------------------------------

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

#----------------------------------------------------------------------------------------------------------------------------------

file(GLOB sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/src/*.cu)

#----------------------------------------------------------------------------------------------------------------------------------

add_library(QTensorNet SHARED ${sources})
target_include_directories(QTensorNet PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(QTensorNet PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(QTensorNet PUBLIC CUDA::cudart 
                                        CUDA::curand
                                        CUDA::cublas                                       
                                        CUDA::cusolver
                                        cutensor 
                                        cutensornet 
                                        MPI::MPI_CXX
                                        yaml-cpp
                                        PkgConfig::SLEPC
                                        PkgConfig::PETSC)

#----------------------------------------------------------------------------------------------------------------------------------

install(TARGETS QTensorNet DESTINATION "${PROJECT_SOURCE_DIR}/install/lib")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include" DESTINATION "${PROJECT_SOURCE_DIR}/install")

#----------------------------------------------------------------------------------------------------------------------------------

add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

#----------------------------------------------------------------------------------------------------------------------------------