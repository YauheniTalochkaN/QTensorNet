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
#include "BaseIntegrator.hh"

namespace QTensorNet
{
    namespace Integrators 
    {
        class LanczosIntegrator : public BaseIntegrator
        {
        public:
            LanczosIntegrator(int kr_size = 30, 
                              double eps = 1.0E-30);
            ~LanczosIntegrator() override;
            void* Integrate(const std::function<void(void*, void*)>& productFunction,
                            int n, 
                            double dt, 
                            const void* initialVector) const override;

        private:
            int kr_size_;
            double eps_;
        };
    }
}