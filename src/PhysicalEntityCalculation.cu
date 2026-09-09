#include "PhysicalEntityCalculation.hh"

namespace QTensorNet
{
    namespace PhysicalEntityCalculation
    {
        double VonNeumannEntropy(const void* rho_matrix, size_t matrix_size)
        {
            cusolverDnHandle_t cusolver_handle;
            HANDLE_CUSOLVER_ERROR(cusolverDnCreate(&cusolver_handle));

            cusolverDnParams_t params;
            HANDLE_CUSOLVER_ERROR(cusolverDnCreateParams(&params));

            cuDoubleComplex* d_rho;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_rho, matrix_size * matrix_size * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_rho, rho_matrix, matrix_size * matrix_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

            cuDoubleComplex* d_eigenvalues;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_eigenvalues, matrix_size * sizeof(cuDoubleComplex)));

            int64_t size = static_cast<int64_t>(matrix_size);

            size_t device_work_size, host_work_size;
            HANDLE_CUSOLVER_ERROR(cusolverDnXgeev_bufferSize(cusolver_handle,
                                                             params,
                                                             CUSOLVER_EIG_MODE_NOVECTOR,
                                                             CUSOLVER_EIG_MODE_NOVECTOR,
                                                             size,
                                                             CUDA_C_64F,
                                                             d_rho,
                                                             size,
                                                             CUDA_C_64F,
                                                             d_eigenvalues,
                                                             CUDA_C_64F,
                                                             nullptr,
                                                             size,
                                                             CUDA_C_64F,
                                                             nullptr,
                                                             size,
                                                             CUDA_C_64F,
                                                             &device_work_size,
                                                             &host_work_size));

            void* d_work;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_work, device_work_size));
            
            void* h_work = malloc(host_work_size);
            
            int* d_info;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_info, sizeof(int)));

            HANDLE_CUSOLVER_ERROR(cusolverDnXgeev(cusolver_handle,
                                                  params,
                                                  CUSOLVER_EIG_MODE_NOVECTOR,
                                                  CUSOLVER_EIG_MODE_NOVECTOR,
                                                  size,
                                                  CUDA_C_64F,
                                                  d_rho,
                                                  size,
                                                  CUDA_C_64F,
                                                  d_eigenvalues,
                                                  CUDA_C_64F,
                                                  nullptr,
                                                  size,
                                                  CUDA_C_64F,
                                                  nullptr,
                                                  size,
                                                  CUDA_C_64F,
                                                  d_work,
                                                  device_work_size,
                                                  h_work,
                                                  host_work_size,
                                                  d_info));

            int h_info = 0;
            HANDLE_CUDA_ERROR(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

            if(h_info != 0) 
            {
                free(h_work);
                HANDLE_CUDA_ERROR(cudaFree(d_rho));
                HANDLE_CUDA_ERROR(cudaFree(d_eigenvalues));
                HANDLE_CUDA_ERROR(cudaFree(d_work));
                HANDLE_CUDA_ERROR(cudaFree(d_info));
                HANDLE_CUSOLVER_ERROR(cusolverDnDestroyParams(params));
                HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(cusolver_handle));

                if(h_info < 0) 
                {
                    throw std::runtime_error("VonNeumannEntropy: " + std::to_string(-h_info) + "-th parameter is invalid");
                } 
                else 
                {
                    throw std::runtime_error("VonNeumannEntropy: QR algorithm failed to converge. " 
                                             + std::to_string(h_info) + " off-diagonal elements did not converge.");
                }
            }

            complexType eigenvalues[matrix_size];
            HANDLE_CUDA_ERROR(cudaMemcpy(eigenvalues, d_eigenvalues, matrix_size * sizeof(complexType), cudaMemcpyDeviceToHost));

            double entropy = 0.0;
            for(size_t i = 0; i < matrix_size; i++) 
            {
                double lambda = std::real(eigenvalues[i]);

                if(lambda > 1.0e-10) 
                {
                    entropy -= lambda * std::log2(lambda);
                }
            }

            free(h_work);
            HANDLE_CUDA_ERROR(cudaFree(d_rho));
            HANDLE_CUDA_ERROR(cudaFree(d_eigenvalues));
            HANDLE_CUDA_ERROR(cudaFree(d_work));
            HANDLE_CUDA_ERROR(cudaFree(d_info));
            HANDLE_CUSOLVER_ERROR(cusolverDnDestroyParams(params));
            HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(cusolver_handle));

            return entropy;
        }
    }
}