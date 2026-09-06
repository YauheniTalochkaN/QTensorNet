#include "CuOperatorMethods.hh"

namespace QTensorNet
{
    namespace CuOperatorMethods
    {
        __global__ void InitRandomVectorKernel(cuDoubleComplex* v, int n, unsigned long long seed) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if(i < n) 
            {
                curandState state;
                curand_init(seed, i, 0, &state);

                double real_part = curand_uniform_double(&state) * 2.0 - 1.0;
                double imag_part = curand_uniform_double(&state) * 2.0 - 1.0;

                v[i] = make_cuDoubleComplex(real_part, imag_part);
            }
        }

        __global__ void FillTmatrixKernel(double* __restrict__ T, const double* __restrict__ alpha, const double* __restrict__ beta, int m) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if(i < m && j < m) 
            {
                double val = 0.0;

                if(i == j) 
                {
                    val = alpha[i];
                }
                else if(i == j + 1) 
                {
                    val = beta[j];
                }
                else if(j == i + 1) 
                {
                    val = beta[i];
                }

                T[i + j * m] = val;
            }
        }

        __global__ void ConvertDoubleVectorToComplexVectorKernel(cuDoubleComplex* __restrict__ vc, const double* __restrict__ vd, int n) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if(i < n) 
            {
                vc[i] = make_cuDoubleComplex(vd[i], 0.0);
            }
        }

        std::pair<complexType, void*> HeOperatorGroundState(const std::variant<void*, std::function<void(void*, void*)>>& A, 
                                                            int n, const void* x0, int kr_size, double eps)
        {
            int m = std::min(n, kr_size);

            cusolverDnHandle_t cusolver_handle;
            HANDLE_CUSOLVER_ERROR(cusolverDnCreate(&cusolver_handle));

            cublasHandle_t cublas_handle;
            HANDLE_CUBLAS_ERROR(cublasCreate(&cublas_handle));

            cudaStream_t stream;
            HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

            HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
            HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

            HANDLE_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream));

            cuDoubleComplex *d_V, *d_w, *d_projections;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_V, n * m * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_w, n * sizeof(cuDoubleComplex)));
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_projections, m * sizeof(cuDoubleComplex)));

            double *d_alpha, *d_beta;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_alpha, m * sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_beta, (m - 1) * sizeof(double)));

            cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0); 
            cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0); 
            cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);

            if(x0 == nullptr)
            {
                int blockSize_1 = 256;
                int numBlocks_1 = (n + blockSize_1 - 1) / blockSize_1;

                auto now_1 = std::chrono::high_resolution_clock::now();
                unsigned long long seed_1 = static_cast<unsigned long long>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now_1.time_since_epoch()).count()
                );

                InitRandomVectorKernel <<< numBlocks_1, blockSize_1, 0, stream >>> (d_V, n, seed_1);
            }
            else
            {
                HANDLE_CUBLAS_ERROR(cublasZcopy(cublas_handle, n, static_cast<const cuDoubleComplex*>(x0), 1, d_V, 1));
            }

            double v0_norm;
            HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_V, 1, &v0_norm));

            cuDoubleComplex inv_v0_norm_c = make_cuDoubleComplex(1.0 / v0_norm, 0.0);
            HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_v0_norm_c, d_V, 1));

            for(int j = 0; j < m; ++j) 
            {
                // w'_j = A * v_j
                if(auto ptr = std::get_if<void*>(&A)) 
                {
                    HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, 
                                                    CUBLAS_OP_N, 
                                                    n, n, 
                                                    &one, 
                                                    static_cast<cuDoubleComplex*>(*ptr), 
                                                    n, d_V + j * n, 1, 
                                                    &zero, 
                                                    d_w, 1));
                } 
                else if(auto func = std::get_if<std::function<void(void*, void*)>>(&A)) 
                {
                    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

                    (*func)(static_cast<void*>(d_V + j * n), static_cast<void*>(d_w));
                }

                // w'_j = w'_j - beta_j * v_j-1
                if(j > 0) 
                {
                    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
                    
                    double b_j;
                    HANDLE_CUDA_ERROR(cudaMemcpy(&b_j, d_beta + j - 1, sizeof(double), cudaMemcpyDeviceToHost));

                    cuDoubleComplex neg_b_j = make_cuDoubleComplex(-b_j, 0.0);
                    HANDLE_CUBLAS_ERROR(cublasZaxpy(cublas_handle, n, &neg_b_j, d_V + (j - 1) * n, 1, d_w, 1));
                }

                // alpha_j = <w'_j | v_j>
                cuDoubleComplex a_j;
                HANDLE_CUBLAS_ERROR(cublasZdotc(cublas_handle, n, d_w, 1, d_V + j * n, 1, &a_j));
                HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_alpha + j, &a_j.x, sizeof(double), cudaMemcpyHostToDevice, stream));

                // w_j = w'_j - alpha_j * v_j
                cuDoubleComplex neg_a_j = make_cuDoubleComplex(-a_j.x, 0.0);
                HANDLE_CUBLAS_ERROR(cublasZaxpy(cublas_handle, n, &neg_a_j, d_V + j * n, 1, d_w, 1));

                // Gram–Schmidt orthogonalization
                HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, CUBLAS_OP_C, n, j + 1, &one, d_V, n, d_w, 1, &zero, d_projections, 1));
                HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, CUBLAS_OP_N, n, j + 1, &neg_one, d_V, n, d_projections, 1, &one, d_w, 1));

                // beta_j+1 = ||w_j||
                double b_jp1;
                HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_w, 1, &b_jp1));

                if(j < m - 1) 
                {
                    if(b_jp1 < eps) 
                    { 
                        b_jp1 = 0.0;

                        int blockSize_2 = 256;
                        int numBlocks_2 = (n + blockSize_2 - 1) / blockSize_2;

                        auto now_2 = std::chrono::high_resolution_clock::now();
                        unsigned long long seed_2 = static_cast<unsigned long long>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(now_2.time_since_epoch()).count()
                        );

                        InitRandomVectorKernel <<< numBlocks_2, blockSize_2, 0, stream >>> (d_w, n, seed_2);

                        double vjp1_norm;
                        HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_w, 1, &vjp1_norm));

                        cuDoubleComplex inv_vjp1_norm_c = make_cuDoubleComplex(1.0 / vjp1_norm, 0.0);
                        HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_vjp1_norm_c, d_w, 1));

                        HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, CUBLAS_OP_C, n, j + 1, &one, d_V, n, d_w, 1, &zero, d_projections, 1));
                        HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, CUBLAS_OP_N, n, j + 1, &neg_one, d_V, n, d_projections, 1, &one, d_w, 1));

                        HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_w, 1, &vjp1_norm));

                        inv_vjp1_norm_c = make_cuDoubleComplex(1.0 / vjp1_norm, 0.0);
                        HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_vjp1_norm_c, d_w, 1));
                    }
                    else
                    {
                        // v_j+1 = w_j / beta_j+1
                        cuDoubleComplex inv_bjp1_c = make_cuDoubleComplex(1.0 / b_jp1, 0.0);
                        HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_bjp1_c, d_w, 1));
                    }

                    HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_beta + j, &b_jp1, sizeof(double), cudaMemcpyHostToDevice, stream));

                    HANDLE_CUBLAS_ERROR(cublasZcopy(cublas_handle, n, d_w, 1, d_V + (j + 1) * n, 1));
                }
            }

            double *d_T, *d_eigenvalues;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_T, m * m * sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_eigenvalues, m * sizeof(double)));

            dim3 block(16, 16);
            dim3 grid((m + block.x - 1) / block.x, (m + block.y - 1) / block.y);

            FillTmatrixKernel <<< grid, block, 0, stream >>> (d_T, d_alpha, d_beta, m);

            int lwork = 0;

            HANDLE_CUSOLVER_ERROR(cusolverDnDsyevd_bufferSize(cusolver_handle, 
                                                              CUSOLVER_EIG_MODE_VECTOR, 
                                                              CUBLAS_FILL_MODE_LOWER, 
                                                              m, 
                                                              d_T, 
                                                              m,
                                                              d_eigenvalues, 
                                                              &lwork));

            double *d_work;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

            int *d_info;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_info, sizeof(int)));

            HANDLE_CUSOLVER_ERROR(cusolverDnDsyevd(cusolver_handle, 
                                                   CUSOLVER_EIG_MODE_VECTOR, 
                                                   CUBLAS_FILL_MODE_LOWER, 
                                                   m, 
                                                   d_T, 
                                                   m, 
                                                   d_eigenvalues, 
                                                   d_work, 
                                                   lwork, 
                                                   d_info));

            int h_info = 0;
            HANDLE_CUDA_ERROR(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

            if(h_info != 0) 
            {            
                HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(cusolver_handle));
                HANDLE_CUBLAS_ERROR(cublasDestroy(cublas_handle));
                HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

                HANDLE_CUDA_ERROR(cudaFree(d_V)); 
                HANDLE_CUDA_ERROR(cudaFree(d_w)); 
                HANDLE_CUDA_ERROR(cudaFree(d_projections));
                HANDLE_CUDA_ERROR(cudaFree(d_alpha)); 
                HANDLE_CUDA_ERROR(cudaFree(d_beta));
                HANDLE_CUDA_ERROR(cudaFree(d_T)); 
                HANDLE_CUDA_ERROR(cudaFree(d_eigenvalues));
                HANDLE_CUDA_ERROR(cudaFree(d_work)); 
                HANDLE_CUDA_ERROR(cudaFree(d_info)); 
            
                throw std::runtime_error("CuOperatorMethods::HeOperatorGroundState: cusolverDnDsyevd failed with info = " + std::to_string(h_info));
            }

            double h_ground_energy;
            HANDLE_CUDA_ERROR(cudaMemcpy(&h_ground_energy, d_eigenvalues, sizeof(double), cudaMemcpyDeviceToHost));

            cuDoubleComplex *d_ground_state;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_ground_state, n * sizeof(cuDoubleComplex)));

            double *d_y_d;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_y_d, m * sizeof(double)));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_y_d, d_T, m * sizeof(double), cudaMemcpyDeviceToDevice));

            cuDoubleComplex *d_y_c;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_y_c, m * sizeof(cuDoubleComplex)));

            int blockSize_3 = 256;
            int numBlocks_3 = (m + blockSize_3 - 1) / blockSize_3;

            ConvertDoubleVectorToComplexVectorKernel <<< numBlocks_3, blockSize_3, 0, stream >>> (d_y_c, d_y_d, m);

            HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, CUBLAS_OP_N, n, m, &one, d_V, n, d_y_c, 1, &zero, d_ground_state, 1));

            double gs_norm;
            HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_ground_state, 1, &gs_norm));

            cuDoubleComplex inv_gs_norm_c = make_cuDoubleComplex(1.0 / gs_norm, 0.0);
            HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_gs_norm_c, d_ground_state, 1));

            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

            HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(cusolver_handle));
            HANDLE_CUBLAS_ERROR(cublasDestroy(cublas_handle));
            HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

            HANDLE_CUDA_ERROR(cudaFree(d_V)); 
            HANDLE_CUDA_ERROR(cudaFree(d_w)); 
            HANDLE_CUDA_ERROR(cudaFree(d_projections));
            HANDLE_CUDA_ERROR(cudaFree(d_alpha)); 
            HANDLE_CUDA_ERROR(cudaFree(d_beta));
            HANDLE_CUDA_ERROR(cudaFree(d_T)); 
            HANDLE_CUDA_ERROR(cudaFree(d_eigenvalues));
            HANDLE_CUDA_ERROR(cudaFree(d_work)); 
            HANDLE_CUDA_ERROR(cudaFree(d_info)); 
            HANDLE_CUDA_ERROR(cudaFree(d_y_d));
            HANDLE_CUDA_ERROR(cudaFree(d_y_c));

            return std::make_pair(complexType(h_ground_energy, 0.0), static_cast<void*>(d_ground_state));
        }

        std::pair<complexType, void*> HeOperatorGroundState(void* A, int n, const void* x0, int max_iter, double eps) 
        {
            return HeOperatorGroundState(std::variant<void*, std::function<void(void*, void*)>>(A),
                                         n, x0, max_iter, eps);
        }

        std::pair<complexType, void*> HeOperatorGroundState(const std::function<void(void*, void*)>& matvec_func, 
                                                            int n, const void* x0, int max_iter, double eps) 
        {
            return HeOperatorGroundState(std::variant<void*, std::function<void(void*, void*)>>(matvec_func),
                                         n, x0, max_iter, eps);
        }
    }
}