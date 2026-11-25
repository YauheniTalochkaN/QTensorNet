#pragma once

#include <iostream>
#include <algorithm>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <complex>
#include <utility>
#include <variant>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#include <petscdevice.h>
#include <slepceps.h>
#include <slepcmfn.h>

#include "CuErrorUtils.hh"
#include "CuHandle.hh"

namespace QTensorNet
{
    namespace SlepcOperatorMethods
    {
        using complexType = std::complex<double>;
        using OperatorVariant = std::variant<const void*, std::function<void(void*, void*)>>;

        void FinalizeSlepc();
        std::pair<complexType, void*> OperatorGroundState(const OperatorVariant& A, 
                                                          int64_t n, 
                                                          const void* x0 = nullptr,
                                                          EPSProblemType problemType = EPS_HEP,
                                                          int kr_size = 30, 
                                                          double tol = 1.0E-10,
                                                          int max_iter = 10000);
        std::pair<complexType, void*> OperatorGroundState(const void* A, 
                                                          int64_t n, 
                                                          const void* x0 = nullptr,
                                                          EPSProblemType problemType = EPS_HEP,
                                                          int kr_size = 30, 
                                                          double tol = 1.0E-10,
                                                          int max_iter = 10000);
        std::pair<complexType, void*> OperatorGroundState(const std::function<void(void*, void*)>& A,
                                                          int64_t n, 
                                                          const void* x0 = nullptr,
                                                          EPSProblemType problemType = EPS_HEP,
                                                          int kr_size = 30, 
                                                          double tol = 1.0E-10,
                                                          int max_iter = 10000);
        void* OperatorExpAction(const OperatorVariant& A,
                                int64_t n,
                                const void* x0,
                                complexType alpha,
                                int kr_size = 30,
                                double tol = 1.0E-10,
                                int max_iter = 10000);
        void* OperatorExpAction(const void* A,
                                int64_t n,
                                const void* x0,
                                complexType alpha,
                                int kr_size = 30,
                                double tol = 1.0E-10,
                                int max_iter = 10000);
        void* OperatorExpAction(const std::function<void(void*, void*)>& A,
                                int64_t n,
                                const void* x0,
                                complexType alpha,
                                int kr_size = 30,
                                double tol = 1.0E-10,
                                int max_iter = 10000);
    }
}