#include "CuHandle.hh"

namespace QTensorNet
{
    namespace CuHandle
    {
        cutensornetHandle_t GetCuTensorNetHandle()
        {
            thread_local cuTensorNetHandleWrapper hw;

            return hw.handle_;
        }

        cusolverDnHandle_t GetCuSolverDnHandle()
        {
            thread_local cuSolverDnHandleWrapper hw;

            return hw.handle_;
        }

        cublasHandle_t GetCuBlasHandle()
        {
            thread_local cuBlasHandleWrapper hw;
            
            return hw.handle_;
        }
    }
}