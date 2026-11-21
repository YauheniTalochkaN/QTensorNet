#pragma once

#include <iostream>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <complex>
#include <utility>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#include <slepceps.h>

#include "CuErrorUtils.hh"
#include "CuHandle.hh"

namespace QTensorNet
{
    namespace SlepcOperatorMethods
    {
        using complexType = std::complex<double>;

        void FinalizeSlepc();
        std::pair<complexType, void*> NonHeOperatorGroundState(const void* A, 
                                                               int n, 
                                                               const void* x0 = nullptr,
                                                               int kr_size = 30, 
                                                               double eps = 1.0E-10,
                                                               int max_iter = 10000);
    }
}