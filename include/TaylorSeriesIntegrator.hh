#pragma once

#include <utility>

#include "BaseIntegrator.hh"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#include "CuErrorUtils.hh"

namespace QTensorNet
{
    namespace Integrators 
    {
        class TaylorSeriesIntegrator : public BaseIntegrator
        {
        public:
            TaylorSeriesIntegrator(double tolerance = 1.0E-6);
            ~TaylorSeriesIntegrator() override;
            void* Integrate(const std::function<void(void*, void*)>& productFunction,
                            int n, 
                            double dt, 
                            const void* initialVector) const override;

        private:
            double tolerance_;
        };
    }
}