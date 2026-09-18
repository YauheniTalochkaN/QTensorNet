#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <complex>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>
#include <cutensornet.h>
#include <mpi.h>

#include "CuErrorUtils.hh"
#include "TensorNetDescriptor.hh"
#include "OperatorVectorProduct.hh"
#include "CuHandle.hh"

namespace QTensorNet
{
    namespace CuTensorNetMethods
    {
        using complexType = std::complex<double>;
        using ContractionOptimizerAttributes = std::vector<std::pair<cutensornetContractionOptimizerConfigAttributes_t, int32_t>>;

        inline const ContractionOptimizerAttributes optimalContractionOptimizerAttributes = {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 0},
                                                                                             {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 0},
                                                                                             {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING, 1},
                                                                                             {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL, 0},
                                                                                             {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM, 0}};
    
        extern bool MPI_;
        extern double flopsToStartMPI_;
    
        extern const cudaDataType_t typeData_; 
        extern const cutensornetComputeType_t typeCompute_;
        
        void SetSVDConfig(cutensornetHandle_t handle,
                          cutensornetTensorSVDConfig_t svdConfig,
                          double absCutoff, 
                          double relCutoff, 
                          cutensornetTensorSVDPartition_t partition,
                          cutensornetTensorSVDNormalization_t renorm,
                          cutensornetTensorSVDAlgo_t svdAlgo,
                          const void* svdParams, 
                          size_t svdParamsSize);
        void SetContractionOptimizerAttributes(cutensornetHandle_t handle,
                                               cutensornetContractionOptimizerConfig_t optimizerConfig,
                                               const ContractionOptimizerAttributes& optimizerAttributes);
        void ContractTensors(const std::vector<std::vector<int32_t>>& modesIn,
                             const std::vector<std::vector<int64_t>>& extentsIn,
                             const std::vector<cutensornetTensorQualifiers_t>& qualifiersIn,
                             const std::vector<const void*>& tensorsIn,
                             const std::vector<int32_t>& modesOut,
                             size_t dimOut,
                             void* tensorOut,
                             cudaStream_t stream,
                             double memoryFactor = 0.8,
                             uint64_t workSpaceLimit = 0UL /*Bytes*/,
                             const std::vector<cutensornetNodePair_t>& pathData = {},
                             const cutensornetWorksizePref_t& workSpacePreference = CUTENSORNET_WORKSIZE_PREF_MIN,
                             const ContractionOptimizerAttributes& optimizerAttributes = optimalContractionOptimizerAttributes,
                             bool mpi = false); 
        void ApplyTensorSVD(const std::vector<int32_t>& modesInAB,
                            const std::vector<int64_t>& extentsInAB,
                            const std::vector<int32_t>& modesOutA,
                            const std::vector<int64_t>& extentsOutA,
                            const std::vector<int32_t>& modesOutB,
                            const std::vector<int64_t>& extentsOutB,
                            const std::pair<size_t, size_t>& bondIndices,
                            const void* tensorInAB,
                            void* tensorOutA,
                            void* tensorOutB,
                            int64_t& newABbondExtent,
                            cudaStream_t stream,
                            double absCutoff, 
                            double relCutoff, 
                            cutensornetTensorSVDPartition_t partition,
                            cutensornetTensorSVDNormalization_t renorm,
                            cutensornetTensorSVDAlgo_t svdAlgo,
                            const void* svdParams, 
                            size_t svdParamsSize,
                            const cutensornetWorksizePref_t& workSpacePreference = CUTENSORNET_WORKSIZE_PREF_MIN,
                            bool verbose = false);
        void ApplyTensorQR(const std::vector<int32_t>& modesInAB,
                           const std::vector<int64_t>& extentsInAB,
                           const std::vector<int32_t>& modesOutA,
                           const std::vector<int64_t>& extentsOutA,
                           const std::vector<int32_t>& modesOutB,
                           const std::vector<int64_t>& extentsOutB,
                           const void* tensorInAB,
                           void* tensorOutA,
                           void* tensorOutB,
                           cudaStream_t stream);
        void SendContractionMetaDataToMPICommWorld(const std::vector<std::vector<int32_t>>& modesIn,
                                                   const std::vector<std::vector<int64_t>>& extentsIn,
                                                   const std::vector<cutensornetTensorQualifiers_t>& qualifiersIn,
                                                   const std::vector<const void*>& tensorsIn,
                                                   const std::vector<int32_t>& modesOut,
                                                   size_t dimOut,
                                                   const cutensornetWorksizePref_t& workSpacePreference,
                                                   const ContractionOptimizerAttributes& optimizerAttributes,
                                                   int32_t numAutotuningIterations);
        void ReceiveContractionMetaDataFromMPICommWorld(std::vector<std::vector<int32_t>>& modesIn,
                                                        std::vector<std::vector<int64_t>>& extentsIn,
                                                        std::vector<cutensornetTensorQualifiers_t>& qualifiersIn,
                                                        std::vector<void*>& tensorsIn,
                                                        std::vector<int32_t>& modesOut,
                                                        size_t& dimOut,
                                                        cutensornetWorksizePref_t& workSpacePreference,
                                                        ContractionOptimizerAttributes& optimizerAttributes,
                                                        int32_t& numAutotuningIterations);
        void SendSignalToMPICommWorld(int32_t key);
        void MPIContractionHelper();
        OperatorVectorProduct* BuildOperatorVectorProduct(const TensorNetDescriptor& rOmegaTNDesc,
                                                          const std::vector<const void*>& rOmegaInTensors,
                                                          cudaStream_t stream,
                                                          double memoryFactor = 0.8,
                                                          uint64_t workSpaceLimit = 0UL /*Bytes*/,
                                                          const cutensornetWorksizePref_t& workSpacePreference = CUTENSORNET_WORKSIZE_PREF_MIN,
                                                          const ContractionOptimizerAttributes& optimizerAttributes = optimalContractionOptimizerAttributes,
                                                          int32_t numAutotuningIterations = 0,
                                                          bool mpi = false);
    }
}