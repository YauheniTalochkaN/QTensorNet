#pragma once

#include <iostream>
#include <functional>

#include "BaseIntegrator.hh"
#include "SlepcOperatorMethods.hh"

namespace QTensorNet
{
    namespace Integrators 
    {
        class KrylovSchurIntegrator : public BaseIntegrator
        {
        public:
            KrylovSchurIntegrator(int kr_size = 30, 
                                  double tol = 1.0E-10,
                                  int max_iter = 10000);
            ~KrylovSchurIntegrator() override;
            void* Integrate(const std::function<void(void*, void*)>& productFunction,
                            int n, 
                            double dt, 
                            const void* initialVector) const override;

        private:
            int kr_size_;
            double tol_;
            int max_iter_;
        };
    }
}