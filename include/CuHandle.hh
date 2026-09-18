#pragma once

#include <cutensornet.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "CuErrorUtils.hh"

namespace QTensorNet
{
    namespace CuHandle
    {
        struct cuTensorNetHandleWrapper
        {
            cutensornetHandle_t handle_{nullptr};

            cuTensorNetHandleWrapper()
            {
                HANDLE_CUTN_ERROR(cutensornetCreate(&handle_));
            }

            ~cuTensorNetHandleWrapper()
            {
                HANDLE_CUTN_ERROR(cutensornetDestroy(handle_));
            }

            cuTensorNetHandleWrapper(const cuTensorNetHandleWrapper&) = delete;
            cuTensorNetHandleWrapper& operator=(const cuTensorNetHandleWrapper&) = delete;
        };

        struct cuSolverDnHandleWrapper
        {
            cusolverDnHandle_t handle_{nullptr};

            cuSolverDnHandleWrapper()
            {
                HANDLE_CUSOLVER_ERROR(cusolverDnCreate(&handle_));
            }

            ~cuSolverDnHandleWrapper()
            {
                HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(handle_));
            }

            cuSolverDnHandleWrapper(const cuSolverDnHandleWrapper&) = delete;
            cuSolverDnHandleWrapper& operator=(const cuSolverDnHandleWrapper&) = delete;
        };

        struct cuBlasHandleWrapper
        {
            cublasHandle_t handle_{nullptr};

            cuBlasHandleWrapper()
            {
                HANDLE_CUBLAS_ERROR(cublasCreate(&handle_));
            }

            ~cuBlasHandleWrapper()
            {
                HANDLE_CUBLAS_ERROR(cublasDestroy(handle_));
            }

            cuBlasHandleWrapper(const cuBlasHandleWrapper&) = delete;
            cuBlasHandleWrapper& operator=(const cuBlasHandleWrapper&) = delete;
        };
        
        cutensornetHandle_t GetCuTensorNetHandle();
        cusolverDnHandle_t GetCuSolverDnHandle();
        cublasHandle_t GetCuBlasHandle();
    }
}