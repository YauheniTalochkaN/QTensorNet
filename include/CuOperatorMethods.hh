#pragma once

#include <iostream>
#include <complex>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "CuErrorUtils.hh"
#include "CuHandle.hh"

namespace QTensorNet
{
    namespace CuOperatorMethods
    {
        using complexType = std::complex<double>;

        double VonNeumannEntropy(const void* A, 
                                 int64_t n);
        void* PseudoInverse(const void* A, 
                            int64_t m, 
                            int64_t n,
                            double threshold = 1.0E-10);
        void* MatMul(const void* A, 
                     const void* B, 
                     int64_t m, 
                     int64_t k, 
                     int64_t n,
                     cublasOperation_t op_A = CUBLAS_OP_N,
                     cublasOperation_t op_B = CUBLAS_OP_N);
        void* MatSum(const void* A,
                     const void* B, 
                     int64_t m,
                     int64_t n,
                     complexType alpha = complexType(1.0, 0.0),
                     complexType beta = complexType(1.0, 0.0),
                     cublasOperation_t op_A = CUBLAS_OP_N,
                     cublasOperation_t op_B = CUBLAS_OP_N);
    }
}