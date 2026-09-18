#pragma once

#include <functional>

namespace QTensorNet
{
    namespace Integrators 
    {
        class BaseIntegrator
        {
        public:
            BaseIntegrator();
            virtual ~BaseIntegrator();
            virtual void* Integrate(const std::function<void(void*, void*)>& productFunction,
                                    int n, 
                                    double dt, 
                                    const void* initialVector) const = 0;
        };
    }
}