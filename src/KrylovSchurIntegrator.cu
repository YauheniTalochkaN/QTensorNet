#include "KrylovSchurIntegrator.hh"

namespace QTensorNet
{
    namespace Integrators 
    {
        KrylovSchurIntegrator::KrylovSchurIntegrator(int kr_size, double tol, int max_iter) : BaseIntegrator()
        {
            kr_size_ = kr_size;
            tol_ = tol;
            max_iter_ = max_iter;
        }

        KrylovSchurIntegrator::~KrylovSchurIntegrator()
        {

        }
        
        void* KrylovSchurIntegrator::Integrate(const std::function<void(void*, void*)>& productFunction,
                                               int n, 
                                               double dt, 
                                               const void* initialVector) const
        {            
            return SlepcOperatorMethods::OperatorExpAction(productFunction, n, initialVector, SlepcOperatorMethods::complexType(0.0, -dt), kr_size_, tol_, max_iter_);
        }
    }
}