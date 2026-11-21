#pragma once

#include <iostream>
#include <utility>
#include <complex>
#include <vector>
#include <chrono>
#include <variant>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include "CuErrorUtils.hh"
#include "CuHandle.hh"

namespace QTensorNet
{
    namespace CuOperatorMethods
    {
        using complexType = std::complex<double>;

        std::pair<complexType, void*> HeOperatorGroundState(const std::variant<const void*, std::function<void(void*, void*)>>& A, 
                                                            int n, 
                                                            const void* x0 = nullptr,
                                                            int kr_size = 30, 
                                                            double eps = 1.0E-30);
        std::pair<complexType, void*> HeOperatorGroundState(const void* A, 
                                                            int n, 
                                                            const void* x0 = nullptr,
                                                            int kr_size = 30, 
                                                            double eps = 1.0E-30);
        std::pair<complexType, void*> HeOperatorGroundState(const std::function<void(void*, void*)>& A, 
                                                            int n, 
                                                            const void* x0 = nullptr,
                                                            int kr_size = 30, 
                                                            double eps = 1.0E-30);
        double VonNeumannEntropy(const void* rho_matrix, 
                                 size_t matrix_size);
        void* PseudoInverse(const void* A, 
                            int m, 
                            int n,
                            double threshold = 1.0E-10);
        void* MatMul(const void* A, 
                     const void* B, 
                     int m, 
                     int k, 
                     int n,
                     cublasOperation_t op_A = CUBLAS_OP_N,
                     cublasOperation_t op_B = CUBLAS_OP_N);
    }
}