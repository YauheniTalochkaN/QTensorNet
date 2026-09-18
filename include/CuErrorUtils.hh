#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cutensornet.h>
#include <cutensor.h>
#include <cusolver_common.h>
#include <cublas_v2.h>
#include <mpi.h>

#define HANDLE_CUDA_ERROR(x) \
{ \
    const auto err = x; \
    if (err != cudaSuccess) \
    { \
        printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
        fflush(stdout); \
        std::exit(1); \
    } \
};

#define HANDLE_CUTEN_ERROR(x) \
{ \
    const auto err = x; \
    if (err != CUTENSOR_STATUS_SUCCESS) \
    { \
        printf("cuTensor error %s in line %d\n", cutensorGetErrorString(err), __LINE__); \
        fflush(stdout); \
        std::exit(1); \
    } \
};

#define HANDLE_CUTN_ERROR(x) \
{ \
    const auto err = x; \
    if (err != CUTENSORNET_STATUS_SUCCESS) \
    { \
        printf("cuTensorNet error %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
        fflush(stdout); \
        std::exit(1); \
    } \
};

inline const char* cusolverGetErrorString(cusolverStatus_t status) 
{
    switch (status) 
    {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
        case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
        case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED: return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
        case CUSOLVER_STATUS_IRS_INTERNAL_ERROR: return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_IRS_NOT_SUPPORTED: return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_IRS_OUT_OF_RANGE: return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
        case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES: return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED: return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED: return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
        case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR: return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
        case CUSOLVER_STATUS_INVALID_WORKSPACE: return "CUSOLVER_STATUS_INVALID_WORKSPACE";
        default: return "UNKNOWN_CUSOLVER_ERROR";
    }
}

#define HANDLE_CUSOLVER_ERROR(x) \
{ \
    const auto err = x; \
    if (err != CUSOLVER_STATUS_SUCCESS) \
    { \
        printf("cuSOLVER error %s in line %d\n", cusolverGetErrorString(err), __LINE__); \
        fflush(stdout); \
        std::exit(1); \
    } \
};

inline const char* cublasGetErrorString(cublasStatus_t status) 
{
    switch (status) 
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN_CUBLAS_ERROR";
    }
}

#define HANDLE_CUBLAS_ERROR(x) \
{ \
    const auto err = x; \
    if (err != CUBLAS_STATUS_SUCCESS) \
    { \
        printf("cuBLAS error %s in line %d\n", cublasGetErrorString(err), __LINE__); \
        fflush(stdout); \
        std::exit(1); \
    } \
};

#define HANDLE_MPI_ERROR(x) \
{ \
    const auto err = x; \
    if (err != MPI_SUCCESS) \
    { \
        char error[MPI_MAX_ERROR_STRING]; \
        int len; \
        MPI_Error_string(err, error, &len); \
        printf("MPI error %s in line %d\n", error, __LINE__); \
        fflush(stdout); \
        MPI_Abort(MPI_COMM_WORLD, err); \
    } \
};
