cmake_minimum_required(VERSION 3.20)

#----------------------------------------------------------------------------------------------------------------------

set(CUDASYS $ENV{CUDASYS})

set(CMAKE_CUDA_TOOLKIT_ROOT_DIR ${CUDASYS})
set(CMAKE_CUDA_COMPILER "${CUDASYS}/bin/nvcc")

link_directories("${CUDASYS}/lib64")

find_package(CUDAToolkit 13.2 REQUIRED COMPONENTS 
             cudart cublas cusparse cufft curand cusolver cusparseLt cudss cufile cupti nccl nvshmem_host)

#----------------------------------------------------------------------------------------------------------------------

set(CAFFE2_USE_CUDNN ON)
set(CAFFE2_USE_CUSPARSELT ON)
set(CAFFE2_USE_CUFILE ON)
set(USE_CUDSS ON)

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "gcc-15")
set(CMAKE_CXX_COMPILER "g++-15")
set(CMAKE_CUDA_HOST_COMPILER "g++-15")

#----------------------------------------------------------------------------------------------------------------------

project(QTensorNet LANGUAGES CXX CUDA)

#----------------------------------------------------------------------------------------------------------------------

find_package(MPI REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

find_package(yaml-cpp REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -fPIC -ffast-math -mavx")

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

#----------------------------------------------------------------------------------------------------------------------

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB TEBD1D_source ${PROJECT_SOURCE_DIR}/TEBD1D.cc)
file(GLOB RK1D_source ${PROJECT_SOURCE_DIR}/RK1D.cc)
file(GLOB TDVP3DSchrodingerComplete_source ${PROJECT_SOURCE_DIR}/TDVP3DSchrodingerComplete.cc)
file(GLOB TDVP3DSchrodingerSquare_source ${PROJECT_SOURCE_DIR}/TDVP3DSchrodingerSquare.cc)
file(GLOB TDVP3DvonNeumann_source ${PROJECT_SOURCE_DIR}/TDVP3DvonNeumann.cc)
file(GLOB DMRGSquareKagome_source ${PROJECT_SOURCE_DIR}/DMRGSquareKagome.cc)
file(GLOB DMRGHoneycomb_source ${PROJECT_SOURCE_DIR}/DMRGHoneycomb.cc)

#----------------------------------------------------------------------------------------------------------------------

add_library(QTensorNet SHARED ${sources})
target_include_directories(QTensorNet PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(QTensorNet PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(QTensorNet PUBLIC CUDA::cudart 
                                        CUDA::cublas 
                                        CUDA::cusparse 
                                        CUDA::cufft 
                                        CUDA::curand
                                        CUDA::cusolver
                                        CUDA::cupti
                                        nccl
                                        nvshmem_host
                                        cudnn
                                        cusparseLt
                                        cudss
                                        cufile
                                        cutensor 
                                        cutensornet 
                                        MPI::MPI_CXX
                                        yaml-cpp)
 
#----------------------------------------------------------------------------------------------------------------------

add_executable(TEBD1D ${TEBD1D_source})
target_compile_features(TEBD1D PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(TEBD1D PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

add_executable(RK1D ${RK1D_source})
target_compile_features(RK1D PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(RK1D PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

add_executable(TDVP3DSchrodingerComplete ${TDVP3DSchrodingerComplete_source})
target_compile_features(TDVP3DSchrodingerComplete PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(TDVP3DSchrodingerComplete PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

add_executable(TDVP3DSchrodingerSquare ${TDVP3DSchrodingerSquare_source})
target_compile_features(TDVP3DSchrodingerSquare PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(TDVP3DSchrodingerSquare PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

add_executable(TDVP3DvonNeumann ${TDVP3DvonNeumann_source})
target_compile_features(TDVP3DvonNeumann PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(TDVP3DvonNeumann PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

add_executable(DMRGSquareKagome ${DMRGSquareKagome_source})
target_compile_features(DMRGSquareKagome PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(DMRGSquareKagome PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

add_executable(DMRGHoneycomb ${DMRGHoneycomb_source})
target_compile_features(DMRGHoneycomb PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(DMRGHoneycomb PRIVATE QTensorNet)

#----------------------------------------------------------------------------------------------------------------------

install(TARGETS TEBD1D 
                RK1D 
                TDVP3DSchrodingerComplete 
                TDVP3DSchrodingerSquare
                TDVP3DvonNeumann 
                DMRGSquareKagome 
                DMRGHoneycomb
                DESTINATION ${PROJECT_SOURCE_DIR}/install)

#----------------------------------------------------------------------------------------------------------------------

add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

#----------------------------------------------------------------------------------------------------------------------