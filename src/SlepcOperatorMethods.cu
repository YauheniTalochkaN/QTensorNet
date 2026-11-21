#include "SlepcOperatorMethods.hh"

namespace
{
    std::once_flag g_slepc_init_flag;
    bool g_slepc_initialized_by_us = false;

    void EnsureSlepcInitialized()
    {
        std::call_once(g_slepc_init_flag, []()
        {
            PetscBool already_init = PETSC_FALSE;
            PetscErrorCode ierr = SlepcInitialized(&already_init);
            if(ierr != PETSC_SUCCESS) { throw std::runtime_error("EnsureSlepcInitialized: SlepcInitialized failed."); }

            if(!already_init)
            {
                ierr = SlepcInitialize(nullptr, nullptr, nullptr, nullptr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("EnsureSlepcInitialized: SlepcInitialize failed."); }

                g_slepc_initialized_by_us = true;
            }
        });
    }
}

namespace QTensorNet
{
    namespace SlepcOperatorMethods
    {        
        void FinalizeSlepc()
        {
            PetscBool is_init = PETSC_FALSE;
            PetscBool is_finalized = PETSC_FALSE;
            if(SlepcInitialized(&is_init) == PETSC_SUCCESS && is_init &&
               PetscFinalized(&is_finalized) == PETSC_SUCCESS && !is_finalized && g_slepc_initialized_by_us)
            {
                SlepcFinalize();
                
                g_slepc_initialized_by_us = false;
            }
        }
        
        std::pair<complexType, void*> NonHeOperatorGroundState(const void* A, int n, const void* x0, int kr_size, double eps, int max_iter)
        {
            if(A == nullptr || n <= 0) 
            {
                throw std::invalid_argument("SlepcOperatorMethods::NonHeOperatorGroundState: invalid matrix pointer or dimension n.");
            }

            EnsureSlepcInitialized();

            PetscInt pn = static_cast<PetscInt>(n);
            int m = std::min(n, std::max(4, kr_size));

            Mat mat_A = nullptr;
            Vec vec_x0 = nullptr;
            Vec vec_xr = nullptr;
            Vec vec_xi = nullptr;
            EPS slepc_eps = nullptr;

            PetscErrorCode ierr = PETSC_SUCCESS;

            try
            {
                ierr = MatCreateDenseCUDA(PETSC_COMM_SELF, pn, pn, pn, pn, nullptr, &mat_A);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: MatCreateDenseCUDA failed."); }

                PetscScalar* d_mat_array = nullptr;
                ierr = MatDenseCUDAGetArrayWrite(mat_A, &d_mat_array);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: MatDenseCUDAGetArrayWrite failed."); }

                HANDLE_CUDA_ERROR(cudaMemcpy(d_mat_array,
                                             static_cast<const cuDoubleComplex*>(A),
                                             static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                             cudaMemcpyDeviceToDevice));

                ierr = MatDenseCUDARestoreArrayWrite(mat_A, &d_mat_array);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: MatDenseCUDARestoreArrayWrite failed."); }

                ierr = MatAssemblyBegin(mat_A, MAT_FINAL_ASSEMBLY);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: MatAssemblyBegin failed."); }
                
                ierr = MatAssemblyEnd(mat_A, MAT_FINAL_ASSEMBLY);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: MatAssemblyEnd failed."); }

                ierr = EPSCreate(PETSC_COMM_SELF, &slepc_eps);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSCreate failed."); }

                ierr = EPSSetOperators(slepc_eps, mat_A, nullptr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetOperators failed."); }

                ierr = EPSSetProblemType(slepc_eps, EPS_NHEP);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetProblemType failed."); }

                ierr = EPSSetType(slepc_eps, EPSKRYLOVSCHUR);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetType failed."); }

                ierr = EPSSetWhichEigenpairs(slepc_eps, EPS_SMALLEST_REAL);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetWhichEigenpairs failed."); }

                ierr = EPSSetDimensions(slepc_eps, 1, m, PETSC_DETERMINE);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetDimensions failed."); }

                ierr = EPSSetTolerances(slepc_eps, eps, max_iter);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetTolerances failed."); }

                if(x0 != nullptr)
                {
                    ierr = VecCreateSeqCUDA(PETSC_COMM_SELF, pn, &vec_x0);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: VecCreateSeqCUDA failed."); }

                    PetscScalar* d_x0_array = nullptr;
                    ierr = VecCUDAGetArrayWrite(vec_x0, &d_x0_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: VecCUDAGetArrayWrite failed."); }

                    HANDLE_CUDA_ERROR(cudaMemcpy(d_x0_array,
                                                 static_cast<const cuDoubleComplex*>(x0),
                                                 static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                                 cudaMemcpyDeviceToDevice));

                    ierr = VecCUDARestoreArrayWrite(vec_x0, &d_x0_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: VecCUDARestoreArrayWrite failed."); }

                    ierr = EPSSetInitialSpace(slepc_eps, 1, &vec_x0);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetInitialSpace failed."); }
                }

                ierr = EPSSetFromOptions(slepc_eps);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSetFromOptions failed."); }

                ierr = EPSSolve(slepc_eps);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSSolve failed."); }

                EPSConvergedReason reason = EPS_CONVERGED_ITERATING;
                ierr = EPSGetConvergedReason(slepc_eps, &reason);
                if(ierr != PETSC_SUCCESS || reason < 0) { throw std::runtime_error("SlepcOperatorMethods::NonHeOperatorGroundState: SLEPc EPS failed to converge."); }

                ierr = MatCreateVecs(mat_A, &vec_xr, &vec_xi);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: MatCreateVecs failed."); }

                PetscScalar eig_val = 0.0;
                ierr = EPSGetEigenpair(slepc_eps, 0, &eig_val, nullptr, vec_xr, vec_xi);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: EPSGetEigenpair failed."); }

                complexType ground_energy(
                    static_cast<double>(PetscRealPart(eig_val)),
                    static_cast<double>(PetscImaginaryPart(eig_val))
                );

                const PetscScalar* d_xr_read = nullptr;
                ierr = VecCUDAGetArrayRead(vec_xr, &d_xr_read);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("NonHeOperatorGroundState: VecCUDAGetArrayRead failed."); }

                cuDoubleComplex* d_ground_state = nullptr;
                HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_ground_state), 
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex)));

                HANDLE_CUDA_ERROR(cudaMemcpy(d_ground_state,
                                             d_xr_read,
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                             cudaMemcpyDeviceToDevice));

                ierr = VecCUDARestoreArrayRead(vec_xr, &d_xr_read);
                if(ierr != PETSC_SUCCESS)
                {
                    HANDLE_CUDA_ERROR(cudaFree(d_ground_state));
                    throw std::runtime_error("NonHeOperatorGroundState: VecCUDARestoreArrayRead failed.");
                }

                cublasHandle_t cublas_handle = CuHandle::GetCuBlasHandle();

                cublasPointerMode_t prev_pointer_mode;
                HANDLE_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_pointer_mode));

                HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

                double gs_norm;
                HANDLE_CUBLAS_ERROR(cublasDznrm2(cublas_handle, n, d_ground_state, 1, &gs_norm));

                cuDoubleComplex inv_norm = make_cuDoubleComplex(1.0 / gs_norm, 0.0);
                HANDLE_CUBLAS_ERROR(cublasZscal(cublas_handle, n, &inv_norm, d_ground_state, 1));

                HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_pointer_mode));

                if(mat_A) { MatDestroy(&mat_A); }
                if(vec_x0) { VecDestroy(&vec_x0); }
                if(vec_xr) { VecDestroy(&vec_xr); }
                if(vec_xi) { VecDestroy(&vec_xi); }
                if(slepc_eps) { EPSDestroy(&slepc_eps); }

                return std::make_pair(ground_energy, static_cast<void*>(d_ground_state));
            }
            catch(...)
            {
                if(mat_A) { MatDestroy(&mat_A); }
                if(vec_x0) { VecDestroy(&vec_x0); }
                if(vec_xr) { VecDestroy(&vec_xr); }
                if(vec_xi) { VecDestroy(&vec_xi); }
                if(slepc_eps) { EPSDestroy(&slepc_eps); }

                throw;
            }
        }
    }
}