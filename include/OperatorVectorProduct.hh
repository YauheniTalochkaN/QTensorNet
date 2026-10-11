#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <complex>

#include <cuda_runtime.h>
#include <cutensornet.h>
#include <mpi.h>

#include "CuErrorUtils.hh"

namespace QTensorNet
{
    class OperatorVectorProduct
    {
    public:
        OperatorVectorProduct(cutensornetHandle_t handle,
                              cudaStream_t stream,
                              MPI_Comm cutnComm,
                              cutensornetNetworkDescriptor_t descNet,
                              int64_t lastTensorID,
                              cutensornetContractionOptimizerConfig_t optimizerConfig,
                              cutensornetContractionOptimizerInfo_t optimizerInfo,
                              cutensornetWorkspaceDescriptor_t workDesc,
                              void* scratchPtr,
                              cutensornetNetworkAutotunePreference_t autotunePref,
                              cutensornetSliceGroup_t sliceGroup,
                              size_t inTensorSize,
                              size_t outTensorSize,
                              bool mpirun);
        ~OperatorVectorProduct();
        OperatorVectorProduct(const OperatorVectorProduct&) = delete;
        OperatorVectorProduct(OperatorVectorProduct&&) = delete;
        OperatorVectorProduct& operator=(const OperatorVectorProduct&) = delete;
        OperatorVectorProduct& operator=(OperatorVectorProduct&&) = delete;
        void DestroyDescNet();
        void DestroyOptimizerConfig();
        void DestroyOptimizerInfo();
        void DestroyWorkDesc();
        void DestroyScratchPtr();
        void DestroyAutotunePref();
        void DestroySliceGroup();
        void ClearTNMetaData();
        const std::function<void(void*, void*)>& GetProductFunction() const;
        size_t GetInTensorSize() const;
        size_t GetOutTensorSize() const;

    private:
        cutensornetHandle_t handle_;
        cudaStream_t stream_;
        MPI_Comm cutnComm_;
        cutensornetNetworkDescriptor_t descNet_;
        int64_t lastTensorID_;
        cutensornetContractionOptimizerConfig_t optimizerConfig_;
        cutensornetContractionOptimizerInfo_t optimizerInfo_;
        cutensornetWorkspaceDescriptor_t workDesc_;
        void *scratchPtr_;
        cutensornetNetworkAutotunePreference_t autotunePref_;
        cutensornetSliceGroup_t sliceGroup_;
        std::function<void(void*, void*)> productFunction_;
        size_t inTensorSize_;
        size_t outTensorSize_;
        bool firstUsage_{true};
        bool mpirun_;
    };
}