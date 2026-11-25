#include "SlepcOperatorMethods.hh"

namespace
{
    std::once_flag g_slepc_init_flag;
    bool g_slepc_initialized_by_us = false;

    struct ShellOperatorContext
    {
        std::function<void(void*, void*)> matvec;
    };

    void EnsureSlepcInitialized()
    {
        std::call_once(g_slepc_init_flag, []()
        {
            PetscBool already_init = PETSC_FALSE;
            PetscErrorCode ierr = SlepcInitialized(&already_init);
            if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::EnsureSlepcInitialized: SlepcInitialized failed."); }

            if(!already_init)
            {
                ierr = SlepcInitialize(nullptr, nullptr, nullptr, nullptr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::EnsureSlepcInitialized: SlepcInitialize failed."); }

                g_slepc_initialized_by_us = true;
            }
        });
    }

    PetscErrorCode MatMultShellCallback(Mat mat, Vec x, Vec y)
    {
        PetscErrorCode ierr = PETSC_SUCCESS;

        void* raw_ctx = nullptr;
        ierr = MatShellGetContext(mat, &raw_ctx);
        if(ierr != PETSC_SUCCESS || raw_ctx == nullptr) { return PETSC_ERR_ARG_WRONGSTATE; }

        auto* ctx = static_cast<ShellOperatorContext*>(raw_ctx);

        const PetscScalar* d_x = nullptr;
        PetscScalar* d_y = nullptr;

        ierr = VecCUDAGetArrayRead(x, &d_x);
        if(ierr != PETSC_SUCCESS) { return ierr; }

        ierr = VecCUDAGetArrayWrite(y, &d_y);
        if(ierr != PETSC_SUCCESS)
        {
            VecCUDARestoreArrayRead(x, &d_x);

            return ierr;
        }

        PetscDeviceContext dctx = nullptr;
        ierr = PetscDeviceContextGetCurrentContext(&dctx);
        if(ierr != PETSC_SUCCESS) 
        { 
            VecCUDARestoreArrayWrite(y, &d_y);
            VecCUDARestoreArrayRead(x, &d_x);

            return ierr; 
        }

        ierr = PetscDeviceContextSynchronize(dctx);
        if(ierr != PETSC_SUCCESS) 
        { 
            VecCUDARestoreArrayWrite(y, &d_y);
            VecCUDARestoreArrayRead(x, &d_x);

            return ierr; 
        }

        try
        {                      
            ctx->matvec(const_cast<void*>(static_cast<const void*>(d_x)), static_cast<void*>(d_y));
        }
        catch(...)
        {
            VecCUDARestoreArrayWrite(y, &d_y);
            VecCUDARestoreArrayRead(x, &d_x);

            return PETSC_ERR_LIB;
        }

        ierr = VecCUDARestoreArrayWrite(y, &d_y);
        if(ierr != PETSC_SUCCESS)
        {
            VecCUDARestoreArrayRead(x, &d_x);

            return ierr;
        }

        ierr = VecCUDARestoreArrayRead(x, &d_x);
        if(ierr != PETSC_SUCCESS) { return ierr; }

        return PETSC_SUCCESS;
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
        
        std::pair<complexType, void*> OperatorGroundState(const OperatorVariant& A, int64_t n, const void* x0, EPSProblemType problemType, int kr_size, double tol, int max_iter)
        {        
            if(problemType != EPS_HEP && problemType != EPS_NHEP)
            {
                throw std::invalid_argument("SlepcOperatorMethods::OperatorGroundState: unsupported EPSProblemType.");
            }
        
            EnsureSlepcInitialized();

            int64_t m = std::min(n, static_cast<int64_t>(kr_size));
        
            Mat mat_A = nullptr;
            Vec vec_x0 = nullptr;
            Vec vec_x = nullptr;
            EPS slepc_eps = nullptr;
            ShellOperatorContext shell_ctx;
        
            PetscErrorCode ierr = PETSC_SUCCESS;
        
            try
            {
                PetscDeviceContext dctx = nullptr;
                ierr = PetscDeviceContextGetCurrentContext(&dctx);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: Failed to get PETSc device context."); }
                
                if(auto ptr = std::get_if<const void*>(&A))
                {
                    ierr = MatCreateDenseCUDA(PETSC_COMM_SELF, n, n, n, n, nullptr, &mat_A);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatCreateDenseCUDA failed."); }
                
                    PetscScalar* d_mat_array = nullptr;
                    ierr = MatDenseCUDAGetArrayWrite(mat_A, &d_mat_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatDenseCUDAGetArrayWrite failed."); }

                    ierr = PetscDeviceContextSynchronize(dctx);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: Failed to synchronize PETSc device context."); }
                
                    HANDLE_CUDA_ERROR(cudaMemcpy(d_mat_array,
                                                 static_cast<const cuDoubleComplex*>(*ptr),
                                                 static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                                 cudaMemcpyDeviceToDevice));
                    
                    ierr = MatDenseCUDARestoreArrayWrite(mat_A, &d_mat_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatDenseCUDARestoreArrayWrite failed."); }
                }
                else if(auto func = std::get_if<std::function<void(void*, void*)>>(&A))
                {
                    shell_ctx.matvec = *func;
                
                    ierr = MatCreateShell(PETSC_COMM_SELF, n, n, n, n, &shell_ctx, &mat_A);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatCreateShell failed."); }
                
                    ierr = MatShellSetOperation(mat_A, MATOP_MULT, reinterpret_cast<void(*)(void)>(MatMultShellCallback));
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatShellSetOperation failed."); }

                    ierr = MatShellSetVecType(mat_A, VECSEQCUDA);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatShellSetVecType failed."); }
                }
            
                if(problemType == EPS_HEP)
                {
                    ierr = MatSetOption(mat_A, MAT_HERMITIAN, PETSC_TRUE);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatSetOption failed."); }
                }
            
                ierr = MatAssemblyBegin(mat_A, MAT_FINAL_ASSEMBLY);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatAssemblyBegin failed."); }
            
                ierr = MatAssemblyEnd(mat_A, MAT_FINAL_ASSEMBLY);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: MatAssemblyEnd failed."); }
            
                ierr = EPSCreate(PETSC_COMM_SELF, &slepc_eps);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSCreate failed."); }
            
                ierr = EPSSetOperators(slepc_eps, mat_A, nullptr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetOperators failed."); }
            
                ierr = EPSSetProblemType(slepc_eps, problemType);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetProblemType failed."); }
            
                ierr = EPSSetType(slepc_eps, EPSKRYLOVSCHUR);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetType failed."); }
            
                ierr = EPSSetWhichEigenpairs(slepc_eps, EPS_SMALLEST_REAL);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetWhichEigenpairs failed."); }
            
                ierr = EPSSetDimensions(slepc_eps, 1, m, PETSC_DETERMINE);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetDimensions failed."); }
            
                ierr = EPSSetTolerances(slepc_eps, tol, max_iter);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetTolerances failed."); }
            
                if(x0 != nullptr)
                {
                    ierr = VecCreateSeqCUDA(PETSC_COMM_SELF, n, &vec_x0);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: VecCreateSeqCUDA(vec_x0) failed."); }
                
                    PetscScalar* d_x0_array = nullptr;
                    ierr = VecCUDAGetArrayWrite(vec_x0, &d_x0_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: VecCUDAGetArrayWrite failed."); }

                    ierr = PetscDeviceContextSynchronize(dctx);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: Failed to synchronize PETSc device context."); }
                
                    HANDLE_CUDA_ERROR(cudaMemcpy(d_x0_array,
                                                 static_cast<const cuDoubleComplex*>(x0),
                                                 static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                                 cudaMemcpyDeviceToDevice));
                    
                    ierr = VecCUDARestoreArrayWrite(vec_x0, &d_x0_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: VecCUDARestoreArrayWrite failed."); }
                    
                    ierr = EPSSetInitialSpace(slepc_eps, 1, &vec_x0);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetInitialSpace failed."); }
                }
            
                ierr = EPSSetFromOptions(slepc_eps);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSetFromOptions failed."); }
            
                ierr = EPSSolve(slepc_eps);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSSolve failed."); }
            
                EPSConvergedReason reason = EPS_CONVERGED_ITERATING;
                ierr = EPSGetConvergedReason(slepc_eps, &reason);
                if(ierr != PETSC_SUCCESS || reason < 0) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: SLEPc EPS failed to converge."); }
            
                ierr = VecCreateSeqCUDA(PETSC_COMM_SELF, n, &vec_x);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: VecCreateSeqCUDA(vec_x) failed."); }
            
                PetscScalar eig_val = 0.0;
                ierr = EPSGetEigenpair(slepc_eps, 0, &eig_val, nullptr, vec_x, nullptr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: EPSGetEigenpair failed."); }

                ierr = PetscDeviceContextSynchronize(dctx);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: Failed to synchronize PETSc device context."); }
            
                complexType ground_energy(
                    static_cast<double>(PetscRealPart(eig_val)),
                    (problemType == EPS_HEP) ? 0.0 : static_cast<double>(PetscImaginaryPart(eig_val))
                );
            
                const PetscScalar* d_x_read = nullptr;
                ierr = VecCUDAGetArrayRead(vec_x, &d_x_read);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: VecCUDAGetArrayRead failed."); }
            
                cuDoubleComplex* d_ground_state = nullptr;
                HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_ground_state),
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex)));
            
                HANDLE_CUDA_ERROR(cudaMemcpy(d_ground_state,
                                             d_x_read,
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                             cudaMemcpyDeviceToDevice));
                
                ierr = VecCUDARestoreArrayRead(vec_x, &d_x_read);
                if(ierr != PETSC_SUCCESS)
                {
                    HANDLE_CUDA_ERROR(cudaFree(d_ground_state));
                    throw std::runtime_error("SlepcOperatorMethods::OperatorGroundState: VecCUDARestoreArrayRead failed.");
                }
            
                cudaStream_t stream = nullptr;
                HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
                
                cublasHandle_t cublas_handle = CuHandle::GetCuBlasHandle();
            
                cudaStream_t prev_cublas_stream = nullptr;
                cublasPointerMode_t prev_pointer_mode;

                HANDLE_CUBLAS_ERROR(cublasGetStream(cublas_handle, &prev_cublas_stream));
                HANDLE_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_pointer_mode));
            
                HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
                HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
            
                double gs_norm;
                HANDLE_CUBLAS_ERROR(cublasDznrm2_64(cublas_handle, n, d_ground_state, 1, &gs_norm));
            
                cuDoubleComplex inv_norm = make_cuDoubleComplex(1.0 / gs_norm, 0.0);
                HANDLE_CUBLAS_ERROR(cublasZscal_64(cublas_handle, n, &inv_norm, d_ground_state, 1));

                HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
            
                HANDLE_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
                HANDLE_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_pointer_mode));

                HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
            
                if(mat_A) { MatDestroy(&mat_A); }
                if(vec_x0) { VecDestroy(&vec_x0); }
                if(vec_x) { VecDestroy(&vec_x); }
                if(slepc_eps) { EPSDestroy(&slepc_eps); }
            
                return std::make_pair(ground_energy, static_cast<void*>(d_ground_state));
            }
            catch(...)
            {
                if(mat_A) { MatDestroy(&mat_A); }
                if(vec_x0) { VecDestroy(&vec_x0); }
                if(vec_x) { VecDestroy(&vec_x); }
                if(slepc_eps) { EPSDestroy(&slepc_eps); }
            
                throw;
            }
        }

        std::pair<complexType, void*> OperatorGroundState(const void* A, int64_t n, const void* x0, EPSProblemType problemType, int kr_size, double tol, int max_iter)
        {
            return OperatorGroundState(OperatorVariant(A), n, x0, problemType, kr_size, tol, max_iter);
        }

        std::pair<complexType, void*> OperatorGroundState(const std::function<void(void*, void*)>& A, int64_t n, const void* x0, EPSProblemType problemType, int kr_size, double tol, int max_iter)
        {
            return OperatorGroundState(OperatorVariant(A), n, x0, problemType, kr_size, tol, max_iter);
        }

        void* OperatorExpAction(const OperatorVariant& A, int64_t n, const void* x0, complexType alpha, int kr_size, double tol, int max_iter)
        {        
            EnsureSlepcInitialized();
        
            int64_t m = std::min(n, static_cast<int64_t>(kr_size));
        
            Mat mat_A = nullptr;
            Vec vec_x = nullptr;
            Vec vec_y = nullptr;
            MFN mfn = nullptr;
            FN fn = nullptr;
            ShellOperatorContext shell_ctx;
        
            PetscErrorCode ierr = PETSC_SUCCESS;
        
            try
            {
                PetscDeviceContext dctx = nullptr;
                ierr = PetscDeviceContextGetCurrentContext(&dctx);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: Failed to get PETSc device context."); }
                
                if(auto ptr = std::get_if<const void*>(&A))
                {
                    ierr = MatCreateDenseCUDA(PETSC_COMM_SELF, n, n, n, n, nullptr, &mat_A);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatCreateDenseCUDA failed."); }
                
                    PetscScalar* d_mat_array = nullptr;
                    ierr = MatDenseCUDAGetArrayWrite(mat_A, &d_mat_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatDenseCUDAGetArrayWrite failed."); }

                    ierr = PetscDeviceContextSynchronize(dctx);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: Failed to synchronize PETSc device context."); }
                
                    HANDLE_CUDA_ERROR(cudaMemcpy(d_mat_array,
                                                 static_cast<const cuDoubleComplex*>(*ptr),
                                                 static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                                 cudaMemcpyDeviceToDevice));
                    
                    ierr = MatDenseCUDARestoreArrayWrite(mat_A, &d_mat_array);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatDenseCUDARestoreArrayWrite failed."); }
                }
                else if(auto func = std::get_if<std::function<void(void*, void*)>>(&A))
                {
                    shell_ctx.matvec = *func;
                
                    ierr = MatCreateShell(PETSC_COMM_SELF, n, n, n, n, &shell_ctx, &mat_A);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatCreateShell failed."); }
                
                    ierr = MatShellSetOperation(mat_A, MATOP_MULT, reinterpret_cast<void(*)(void)>(MatMultShellCallback));
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatShellSetOperation failed."); }

                    ierr = MatShellSetVecType(mat_A, VECSEQCUDA);
                    if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatShellSetVecType failed."); }
                }
            
                ierr = MatAssemblyBegin(mat_A, MAT_FINAL_ASSEMBLY);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatAssemblyBegin failed."); }
            
                ierr = MatAssemblyEnd(mat_A, MAT_FINAL_ASSEMBLY);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MatAssemblyEnd failed."); }
            
                ierr = VecCreateSeqCUDA(PETSC_COMM_SELF, n, &vec_x);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: VecCreateSeqCUDA(vec_x) failed."); }
            
                PetscScalar* d_x_arr = nullptr;
                ierr = VecCUDAGetArrayWrite(vec_x, &d_x_arr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: VecCUDAGetArrayWrite failed."); }

                ierr = PetscDeviceContextSynchronize(dctx);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: Failed to synchronize PETSc device context."); }
            
                HANDLE_CUDA_ERROR(cudaMemcpy(d_x_arr,
                                             static_cast<const cuDoubleComplex*>(x0),
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                             cudaMemcpyDeviceToDevice));
                
                ierr = VecCUDARestoreArrayWrite(vec_x, &d_x_arr);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: VecCUDARestoreArrayWrite failed."); }
                
                ierr = VecCreateSeqCUDA(PETSC_COMM_SELF, n, &vec_y);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: VecCreateSeqCUDA(vec_y) failed."); }
                
                ierr = MFNCreate(PETSC_COMM_SELF, &mfn);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNCreate failed."); }
                
                ierr = MFNSetOperator(mfn, mat_A);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNSetOperator failed."); }
                
                ierr = MFNGetFN(mfn, &fn);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNGetFN failed."); }
                
                ierr = FNSetType(fn, FNEXP);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: FNSetType(FNEXP) failed."); }
                
                PetscScalar petsc_alpha = static_cast<PetscScalar>(alpha);
                ierr = FNSetScale(fn, petsc_alpha, static_cast<PetscScalar>(1.0));
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: FNSetScale failed."); }
                
                ierr = MFNSetDimensions(mfn, m);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNSetDimensions failed."); }
                
                ierr = MFNSetTolerances(mfn, tol, max_iter);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNSetTolerances failed."); }
                
                ierr = MFNSetFromOptions(mfn);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNSetFromOptions failed."); }
                
                ierr = MFNSolve(mfn, vec_x, vec_y);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: MFNSolve failed."); }
                
                MFNConvergedReason reason = MFN_CONVERGED_ITERATING;
                ierr = MFNGetConvergedReason(mfn, &reason);
                if(ierr != PETSC_SUCCESS || reason < 0) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: SLEPc MFN failed to converge."); }
            
                const PetscScalar* d_y_read = nullptr;
                ierr = VecCUDAGetArrayRead(vec_y, &d_y_read);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: VecCUDAGetArrayRead failed."); }

                ierr = PetscDeviceContextSynchronize(dctx);
                if(ierr != PETSC_SUCCESS) { throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: Failed to synchronize PETSc device context."); }
            
                cuDoubleComplex* d_result = nullptr;
                HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_result),
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex)));
            
                HANDLE_CUDA_ERROR(cudaMemcpy(d_result,
                                             d_y_read,
                                             static_cast<size_t>(n) * sizeof(cuDoubleComplex),
                                             cudaMemcpyDeviceToDevice));
                
                ierr = VecCUDARestoreArrayRead(vec_y, &d_y_read);
                if(ierr != PETSC_SUCCESS)
                {
                    HANDLE_CUDA_ERROR(cudaFree(d_result));
                    throw std::runtime_error("SlepcOperatorMethods::OperatorExpAction: VecCUDARestoreArrayRead failed.");
                }
            
                if(mfn) { MFNDestroy(&mfn); }
                if(vec_y) { VecDestroy(&vec_y); }
                if(vec_x) { VecDestroy(&vec_x); }
                if(mat_A) { MatDestroy(&mat_A); }
            
                return static_cast<void*>(d_result);
            }
            catch(...)
            {
                if(mfn) { MFNDestroy(&mfn); }
                if(vec_y) { VecDestroy(&vec_y); }
                if(vec_x) { VecDestroy(&vec_x); }
                if(mat_A) { MatDestroy(&mat_A); }

                throw;
            }
        }

        void* OperatorExpAction(const void* A, int64_t n, const void* x0, complexType alpha, int kr_size, double tol, int max_iter)
        {
            return OperatorExpAction(OperatorVariant(A), n, x0, alpha, kr_size, tol, max_iter);
        }

        void* OperatorExpAction(const std::function<void(void*, void*)>& A, int64_t n, const void* x0, complexType alpha, int kr_size, double tol, int max_iter)
        {
            return OperatorExpAction(OperatorVariant(A), n, x0, alpha, kr_size, tol, max_iter);
        }
    }
}