#include "TaylorSeriesIntegrator.hh"

namespace QTensorNet
{
    namespace Integrators 
    {
        TaylorSeriesIntegrator::TaylorSeriesIntegrator(double tolerance) : BaseIntegrator()
        {
            tolerance_ = tolerance;
        }

        TaylorSeriesIntegrator::~TaylorSeriesIntegrator()
        {

        }
        
        void* TaylorSeriesIntegrator::Integrate(const std::function<void(void*, void*)>& productFunction,
                                                int n, 
                                                double dt, 
                                                const void* initialVector) const
        {            
            cublasHandle_t cublas_handle;
            HANDLE_CUBLAS_ERROR(cublasCreate(&cublas_handle));

            cudaStream_t stream;
            HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
            
            cuDoubleComplex* result;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&result, n * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMemcpy(result, initialVector, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

            cuDoubleComplex* vec_pre;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&vec_pre, n * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMemcpy(vec_pre, initialVector, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

            cuDoubleComplex* vec_post;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&vec_post, n * sizeof(cuDoubleComplex)));

            cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0); 

            double iter = 1.0;

            while(true)
            {
                productFunction(static_cast<void*>(vec_pre), static_cast<void*>(vec_post));

                cuDoubleComplex coeff = make_cuDoubleComplex(0.0, -dt / iter);
                HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &coeff, vec_post, 1));

                HANDLE_CUBLAS_ERROR(cublasZaxpy(cublas_handle, n, &one, vec_post, 1, result, 1));

                double dv_norm;
                HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, vec_post, 1, &dv_norm));

                std::swap(vec_pre, vec_post);

                double result_norm;
                HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, result, 1, &result_norm));

                if(tolerance_ > dv_norm / result_norm)
                {
                    break;
                }
                
                ++iter;
            }

            HANDLE_CUBLAS_ERROR(cublasDestroy(cublas_handle));
            HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

            HANDLE_CUDA_ERROR(cudaFree(vec_pre)); 
            HANDLE_CUDA_ERROR(cudaFree(vec_post)); 

            return static_cast<void*>(result);
        }
    }
}