#include "LanczosIntegrator.hh"

namespace QTensorNet
{
    namespace Integrators 
    {
        LanczosIntegrator::LanczosIntegrator(int kr_size, double eps) : BaseIntegrator()
        {
            kr_size_ = kr_size;
            eps_ = eps;
        }

        LanczosIntegrator::~LanczosIntegrator()
        {

        }

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

        __global__ void ComputeComplexExpCoeffKernel(cuDoubleComplex* c, const double* eig, const double* T, int m, double dt) 
        {
            int j = blockIdx.x * blockDim.x + threadIdx.x;

            if(j < m) 
            {
                double arg = -eig[j] * dt;

                double cos_val = cos(arg);
                double sin_val = sin(arg);

                double s0j = T[j * m]; 

                c[j] = make_cuDoubleComplex(cos_val * s0j, sin_val * s0j);
            }
        }
        
        void* LanczosIntegrator::Integrate(const std::function<void(void*, void*)>& productFunction,
                                           int n, 
                                           double dt, 
                                           const void* initialVector) const
        {            
            int m = std::min(n, kr_size_);

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

            HANDLE_CUBLAS_ERROR(cublasZcopy(cublas_handle, n, static_cast<const cuDoubleComplex*>(initialVector), 1, d_V, 1));

            double v0_norm;
            HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_V, 1, &v0_norm));

            cuDoubleComplex inv_v0_norm_c = make_cuDoubleComplex(1.0 / v0_norm, 0.0);
            HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_v0_norm_c, d_V, 1));

            for(int j = 0; j < m; ++j) 
            {
                // w'_j = A * v_j
                HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
                
                productFunction(static_cast<void*>(d_V + j * n), static_cast<void*>(d_w));

                // w'_j = w'_j - beta_j * v_j-1
                if(j > 0) 
                {                    
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
                    if(b_jp1 < eps_) 
                    { 
                        b_jp1 = 0.0;

                        int blockSize = 256;
                        int numBlocks = (n + blockSize - 1) / blockSize;

                        auto now = std::chrono::high_resolution_clock::now();
                        unsigned long long seed = static_cast<unsigned long long>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()
                        );

                        InitRandomVectorKernel <<< numBlocks, blockSize, 0, stream >>> (d_w, n, seed);

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
            
                throw std::runtime_error("LanczosIntegrator::Integrate: cusolverDnDsyevd failed with info = " + std::to_string(h_info));
            }

            cuDoubleComplex *d_c;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_c, m * sizeof(cuDoubleComplex)));

            int blockSize = 256;
            int numBlocks = (m + blockSize - 1) / blockSize;

            // c_j = exp(-i dt theta_j) T_0,j
            ComputeComplexExpCoeffKernel <<< numBlocks, blockSize, 0, stream >>> (d_c, d_eigenvalues, d_T, m, dt);

            cuDoubleComplex *d_y;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_y, m * sizeof(cuDoubleComplex)));

            double* d_y_real = reinterpret_cast<double*>(d_y);
            double* d_y_imag = reinterpret_cast<double*>(d_y) + 1;

            double* d_c_real = reinterpret_cast<double*>(d_c);
            double* d_c_imag = reinterpret_cast<double*>(d_c) + 1;

            // y = v_0_norm * T * c
            HANDLE_CUBLAS_ERROR(cublasDgemv(cublas_handle, CUBLAS_OP_N, m, m, &v0_norm, d_T, m, d_c_real, 2, &zero.x, d_y_real, 2));
            HANDLE_CUBLAS_ERROR(cublasDgemv(cublas_handle, CUBLAS_OP_N, m, m, &v0_norm, d_T, m, d_c_imag, 2, &zero.x, d_y_imag, 2));
            
            // result = V * y
            cuDoubleComplex *result;
            HANDLE_CUDA_ERROR(cudaMalloc((void**)&result, n * sizeof(cuDoubleComplex)));

            HANDLE_CUBLAS_ERROR(cublasZgemv(cublas_handle, CUBLAS_OP_N, n, m, &one, d_V, n, d_y, 1, &zero, result, 1));

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
            HANDLE_CUDA_ERROR(cudaFree(d_c));
            HANDLE_CUDA_ERROR(cudaFree(d_y));

            return static_cast<void*>(result);
        }
    }
}