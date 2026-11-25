#include "CuOperatorMethods.hh"

namespace QTensorNet
{
    namespace CuOperatorMethods
    {
        double VonNeumannEntropy(const void* A, int64_t n)
        {
            cusolverDnHandle_t cusolver_handle = CuHandle::GetCuSolverDnHandle();

            cusolverDnParams_t params;
            HANDLE_CUSOLVER_ERROR(cusolverDnCreateParams(&params));

            cuDoubleComplex* d_A;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_A, n * n * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_A, A, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

            cuDoubleComplex* d_eigenvalues;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_eigenvalues, n * sizeof(cuDoubleComplex)));

            size_t device_work_size, host_work_size;
            HANDLE_CUSOLVER_ERROR(cusolverDnXgeev_bufferSize(cusolver_handle,
                                                             params,
                                                             CUSOLVER_EIG_MODE_NOVECTOR,
                                                             CUSOLVER_EIG_MODE_NOVECTOR,
                                                             n,
                                                             CUDA_C_64F,
                                                             d_A,
                                                             n,
                                                             CUDA_C_64F,
                                                             d_eigenvalues,
                                                             CUDA_C_64F,
                                                             nullptr,
                                                             n,
                                                             CUDA_C_64F,
                                                             nullptr,
                                                             n,
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
                                                  n,
                                                  CUDA_C_64F,
                                                  d_A,
                                                  n,
                                                  CUDA_C_64F,
                                                  d_eigenvalues,
                                                  CUDA_C_64F,
                                                  nullptr,
                                                  n,
                                                  CUDA_C_64F,
                                                  nullptr,
                                                  n,
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
                HANDLE_CUDA_ERROR(cudaFree(d_A));
                HANDLE_CUDA_ERROR(cudaFree(d_eigenvalues));
                HANDLE_CUDA_ERROR(cudaFree(d_work));
                HANDLE_CUDA_ERROR(cudaFree(d_info));
                HANDLE_CUSOLVER_ERROR(cusolverDnDestroyParams(params));

                if(h_info < 0) 
                {
                    throw std::runtime_error("VonNeumannEntropy: " + std::to_string(-h_info) + "-th parameter is invalid.");
                } 
                else 
                {
                    throw std::runtime_error("VonNeumannEntropy: QR algorithm failed to converge. " 
                                             + std::to_string(h_info) + " off-diagonal elements did not converge.");
                }
            }

            std::vector<complexType> eigenvalues(n);
            HANDLE_CUDA_ERROR(cudaMemcpy(eigenvalues.data(), d_eigenvalues, n * sizeof(complexType), cudaMemcpyDeviceToHost));

            double entropy = 0.0;

            for(const complexType& el : eigenvalues) 
            {
                double lambda = std::real(el);

                if(lambda > 1.0e-10) 
                {
                    entropy -= lambda * std::log2(lambda);
                }
            }

            free(h_work);
            HANDLE_CUDA_ERROR(cudaFree(d_A));
            HANDLE_CUDA_ERROR(cudaFree(d_eigenvalues));
            HANDLE_CUDA_ERROR(cudaFree(d_work));
            HANDLE_CUDA_ERROR(cudaFree(d_info));
            HANDLE_CUSOLVER_ERROR(cusolverDnDestroyParams(params));

            return entropy;
        }

        __global__ void ComputeInverseSingularValuesKernel(cuDoubleComplex* __restrict__ inv_s, const double* __restrict__ s, int64_t n, double threshold) 
        {
            int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        
            if(i < n) 
            {
                double inv = (s[i] > threshold * s[0]) ? (1.0 / s[i]) : 0.0;
            
                inv_s[i] = make_cuDoubleComplex(inv, 0.0);
            }
        }

        void* PseudoInverse(const void* A, int64_t m, int64_t n, double threshold)
        {
            cusolverDnHandle_t cusolver_handle = CuHandle::GetCuSolverDnHandle();
            cublasHandle_t cublas_handle = CuHandle::GetCuBlasHandle();

            cudaStream_t prev_cublas_stream = nullptr;
            cudaStream_t prev_cusolver_stream = nullptr;
            cublasPointerMode_t prev_pointer_mode;

            HANDLE_CUBLAS_ERROR(cublasGetStream(cublas_handle, &prev_cublas_stream));
            HANDLE_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_pointer_mode));
            HANDLE_CUSOLVER_ERROR(cusolverDnGetStream(cusolver_handle, &prev_cusolver_stream));

            cudaStream_t stream;
            HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
            HANDLE_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream));

            cusolverDnParams_t params;
            HANDLE_CUSOLVER_ERROR(cusolverDnCreateParams(&params));

            int64_t min_mn = std::min(m, n);

            cuDoubleComplex* d_A;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_A, m * n * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_A, A, m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream));

            double* d_S;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_S, min_mn * sizeof(double)));

            cuDoubleComplex* d_U;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_U, m * min_mn * sizeof(cuDoubleComplex)));

            cuDoubleComplex* d_VH;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_VH, min_mn * n * sizeof(cuDoubleComplex)));

            size_t device_work_size, host_work_size;
            HANDLE_CUSOLVER_ERROR(cusolverDnXgesvd_bufferSize(cusolver_handle,
                                                              params,
                                                              'S',
                                                              'S',
                                                              m,
                                                              n,
                                                              CUDA_C_64F,
                                                              d_A,
                                                              m,
                                                              CUDA_R_64F,
                                                              d_S,
                                                              CUDA_C_64F,
                                                              d_U,
                                                              m,
                                                              CUDA_C_64F,
                                                              d_VH,
                                                              min_mn,
                                                              CUDA_C_64F,
                                                              &device_work_size,
                                                              &host_work_size));

            void* d_work;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_work, device_work_size));
            
            void* h_work = malloc(host_work_size);
            
            int* d_info;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_info, sizeof(int)));

            HANDLE_CUSOLVER_ERROR(cusolverDnXgesvd(cusolver_handle,
                                                   params,
                                                   'S',
                                                   'S',
                                                   m,
                                                   n,
                                                   CUDA_C_64F,
                                                   d_A,
                                                   m,
                                                   CUDA_R_64F,
                                                   d_S,
                                                   CUDA_C_64F,
                                                   d_U,
                                                   m,
                                                   CUDA_C_64F,
                                                   d_VH,
                                                   min_mn,
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
                HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
                HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_pointer_mode));
                HANDLE_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, prev_cusolver_stream));

                HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

                free(h_work);
                HANDLE_CUDA_ERROR(cudaFree(d_A));
                HANDLE_CUDA_ERROR(cudaFree(d_S));
                HANDLE_CUDA_ERROR(cudaFree(d_U));
                HANDLE_CUDA_ERROR(cudaFree(d_VH));
                HANDLE_CUDA_ERROR(cudaFree(d_work));
                HANDLE_CUDA_ERROR(cudaFree(d_info));
                HANDLE_CUSOLVER_ERROR(cusolverDnDestroyParams(params));

                throw std::runtime_error("PseudoInverse: cusolverDnZgesvd failed with info = " + std::to_string(h_info));
            }

            cuDoubleComplex* d_inv_s;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_inv_s, min_mn * sizeof(cuDoubleComplex)));

            int64_t blockSize = 256;
            int64_t numBlocks = (min_mn + blockSize - 1) / blockSize;

            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

            ComputeInverseSingularValuesKernel <<< numBlocks, blockSize, 0, stream >>> (d_inv_s, d_S, min_mn, threshold);

            HANDLE_CUBLAS_ERROR(cublasZdgmm_64(cublas_handle,
                                               CUBLAS_SIDE_RIGHT,
                                               m, 
                                               min_mn,
                                               d_U, 
                                               m,
                                               d_inv_s, 
                                               1,
                                               d_U, 
                                               m));

            cuDoubleComplex* d_A_plus;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_A_plus, n * m * sizeof(cuDoubleComplex)));

            HANDLE_CUBLAS_ERROR(cublasZgemm_64(cublas_handle,
                                               CUBLAS_OP_C, 
                                               CUBLAS_OP_C,
                                               n, 
                                               m, 
                                               min_mn,
                                               &alpha,
                                               d_VH, 
                                               min_mn,
                                               d_U, 
                                               m,
                                               &beta,
                                               d_A_plus, 
                                               n));

            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_pointer_mode));
            HANDLE_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, prev_cusolver_stream));

            HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

            free(h_work);
            HANDLE_CUDA_ERROR(cudaFree(d_A));
            HANDLE_CUDA_ERROR(cudaFree(d_S));
            HANDLE_CUDA_ERROR(cudaFree(d_U));
            HANDLE_CUDA_ERROR(cudaFree(d_VH));
            HANDLE_CUDA_ERROR(cudaFree(d_work));
            HANDLE_CUDA_ERROR(cudaFree(d_info));
            HANDLE_CUDA_ERROR(cudaFree(d_inv_s));
            HANDLE_CUSOLVER_ERROR(cusolverDnDestroyParams(params));

            return static_cast<void*>(d_A_plus);
        }

        void* MatMul(const void* A, const void* B, int64_t m, int64_t k, int64_t n, cublasOperation_t op_A, cublasOperation_t op_B)
        {
            cublasHandle_t cublas_handle = CuHandle::GetCuBlasHandle();

            cudaStream_t prev_cublas_stream = nullptr;
            cublasPointerMode_t prev_pointer_mode;

            HANDLE_CUBLAS_ERROR(cublasGetStream(cublas_handle, &prev_cublas_stream));
            HANDLE_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_pointer_mode));

            cudaStream_t stream;
            HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

            int64_t lda = (op_A == CUBLAS_OP_N) ? m : k;
            int64_t ldb = (op_B == CUBLAS_OP_N) ? k : n;

            cuDoubleComplex* d_C;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_C, m * n * sizeof(cuDoubleComplex)));

            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

            HANDLE_CUBLAS_ERROR(cublasZgemm_64(cublas_handle,
                                               op_A, 
                                               op_B,
                                               m, 
                                               n, 
                                               k,
                                               &alpha,
                                               static_cast<const cuDoubleComplex*>(A), 
                                               lda,
                                               static_cast<const cuDoubleComplex*>(B), 
                                               ldb,
                                               &beta,
                                               d_C, 
                                               m));

            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_pointer_mode));

            HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

            return static_cast<void*>(d_C);
        }

        void* MatSum(const void* A, const void* B, int64_t m, int64_t n, complexType alpha, complexType beta, cublasOperation_t op_A, cublasOperation_t op_B)
        {
            cublasHandle_t cublas_handle = CuHandle::GetCuBlasHandle();
        
            cudaStream_t prev_cublas_stream = nullptr;
            cublasPointerMode_t prev_pointer_mode;
        
            HANDLE_CUBLAS_ERROR(cublasGetStream(cublas_handle, &prev_cublas_stream));
            HANDLE_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_pointer_mode));
        
            cudaStream_t stream;
            HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
        
            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
        
            int64_t lda = (op_A == CUBLAS_OP_N) ? m : n;
            int64_t ldb = (op_B == CUBLAS_OP_N) ? m : n;
        
            cuDoubleComplex* d_C;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_C, m * n * sizeof(cuDoubleComplex)));
        
            cuDoubleComplex cu_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
            cuDoubleComplex cu_beta  = make_cuDoubleComplex(beta.real(), beta.imag());
        
            HANDLE_CUBLAS_ERROR(cublasZgeam_64(cublas_handle,
                                               op_A,
                                               op_B,
                                               m,
                                               n,
                                               &cu_alpha,
                                               static_cast<const cuDoubleComplex*>(A),
                                               lda,
                                               &cu_beta,
                                               static_cast<const cuDoubleComplex*>(B),
                                               ldb,
                                               d_C,
                                               m));
            
            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
            
            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_pointer_mode));
            
            HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
            
            return static_cast<void*>(d_C);
        }
    }
}