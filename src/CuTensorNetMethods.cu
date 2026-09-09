#include "CuTensorNetMethods.hh"

namespace QTensorNet
{
    namespace CuTensorNetMethods
    {
        bool MPI_ = false;
        double flopsToStartMPI_ = 1.0E14;
        
        const cudaDataType_t typeData_ = CUDA_C_64F; 
        const cutensornetComputeType_t typeCompute_ = CUTENSORNET_COMPUTE_64F;

        void SetContractionOptimizerAttributes(cutensornetHandle_t handle,
                                               cutensornetContractionOptimizerConfig_t optimizerConfig,
                                               const ContractionOptimizerAttributes& optimizerAttributes)
        {
            for(const auto& [attr, val] : optimizerAttributes)
            {
                switch(attr)
                {
                    case CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM:
                    {
                        auto enumVal = static_cast<cutensornetGraphAlgo_t>(val);

                        HANDLE_CUTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                                            optimizerConfig,
                                                                                            attr,
                                                                                            &enumVal,
                                                                                            sizeof(enumVal)));
                    }
                    break;
                
                    case CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL:
                    {
                        auto enumVal = static_cast<cutensornetMemoryModel_t>(val);

                        HANDLE_CUTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                                            optimizerConfig,
                                                                                            attr,
                                                                                            &enumVal,
                                                                                            sizeof(enumVal)));
                    }
                    break;
                
                    case CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE:
                    {
                        auto enumVal = static_cast<cutensornetOptimizerCost_t>(val);

                        HANDLE_CUTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                                            optimizerConfig,
                                                                                            attr,
                                                                                            &enumVal,
                                                                                            sizeof(enumVal)));
                    }
                    break;
                
                    case CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION:
                    {
                        auto enumVal = static_cast<cutensornetSmartOption_t>(val);

                        HANDLE_CUTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                                            optimizerConfig,
                                                                                            attr,
                                                                                            &enumVal,
                                                                                            sizeof(enumVal)));
                    }
                    break;
                
                    default:

                        HANDLE_CUTN_ERROR(cutensornetContractionOptimizerConfigSetAttribute(handle,
                                                                                            optimizerConfig,
                                                                                            attr,
                                                                                            &val,
                                                                                            sizeof(val)));
                    break;
                }
            }
        }

        void ContractTensors(cutensornetHandle_t handle,
                             cudaStream_t stream,
                             const std::vector<std::vector<int32_t>>& modesIn,
                             const std::vector<std::vector<int64_t>>& extentsIn,
                             const std::vector<cutensornetTensorQualifiers_t>& qualifiersIn,
                             const std::vector<const void*>& tensorsIn,
                             const std::vector<int32_t>& modesOut,
                             size_t dimOut,
                             void* tensorOut,
                             double memoryFactor,
                             uint64_t workSpaceLimit,
                             const std::vector<cutensornetNodePair_t>& pathData,
                             const cutensornetWorksizePref_t& workSpacePreference,
                             const ContractionOptimizerAttributes& optimizerAttributes,
                             bool mpi)
        {        
            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            cutensornetNetworkDescriptor_t descNet;
            HANDLE_CUTN_ERROR(cutensornetCreateNetwork(handle, &descNet));

            size_t numInputTensors = modesIn.size();

            std::vector<int64_t> tensorIDs(numInputTensors);

            for(size_t t = 0; t < numInputTensors; ++t)
            {
                HANDLE_CUTN_ERROR(cutensornetNetworkAppendTensor(handle,
                                                                 descNet,
                                                                 static_cast<int32_t>(modesIn[t].size()),
                                                                 extentsIn[t].data(),
                                                                 modesIn[t].data(),
                                                                 &qualifiersIn[t],
                                                                 typeData_,
                                                                 &tensorIDs[t]));
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkSetOutputTensor(handle,
                                                                descNet,
                                                                static_cast<int32_t>(modesOut.size()),
                                                                (modesOut.size() > 0UL) ? modesOut.data() : nullptr,
                                                                typeData_));


            HANDLE_CUTN_ERROR(cutensornetNetworkSetAttribute(handle,
                                                             descNet,
                                                             CUTENSORNET_NETWORK_COMPUTE_TYPE,
                                                             &typeCompute_,
                                                             sizeof(typeCompute_)));

            size_t freeMem, totalMem;
            HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
            uint64_t limit = static_cast<uint64_t>(static_cast<double>(freeMem) * memoryFactor);

            if((limit > workSpaceLimit) && (workSpaceLimit > 0UL))
            {
                limit = workSpaceLimit;
            }

            bool mpirun = false;
            MPI_Comm cutnComm = MPI_COMM_NULL;

            if((rank > 0) && mpi && MPI_)
            {
                mpirun = true;

                HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm));
                HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle, &cutnComm, sizeof(cutnComm)));
            }

            cutensornetContractionOptimizerInfo_t optimizerInfo;
            cutensornetContractionOptimizerConfig_t optimizerConfig;

            HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

            if(pathData.empty())
            {
                HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));
                SetContractionOptimizerAttributes(handle, optimizerConfig, optimizerAttributes);
                
                HANDLE_CUTN_ERROR(cutensornetContractionOptimize(handle, 
                                                                 descNet, 
                                                                 optimizerConfig, 
                                                                 limit, 
                                                                 optimizerInfo));
            }
            else
            {
                cutensornetContractionPath_t customPath;

                customPath.numContractions = static_cast<int32_t>(pathData.size());
                customPath.data = const_cast<cutensornetNodePair_t*>(pathData.data());

                HANDLE_CUTN_ERROR(cutensornetContractionOptimizerInfoSetAttribute(handle, 
                                                                                  optimizerInfo, 
                                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, 
                                                                                  &customPath, 
                                                                                  sizeof(customPath)));

                HANDLE_CUTN_ERROR(cutensornetNetworkSetOptimizerInfo(handle,
                                                                     descNet,
                                                                     optimizerInfo));
            }
            
            if((rank == 0) && mpi && MPI_ && pathData.empty())
            {         
                double flopCount;
                HANDLE_CUTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                                  optimizerInfo,
                                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                                  &flopCount,
                                                                                  sizeof(flopCount)));
                
                int64_t sliceNumber;
                HANDLE_CUTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                                  optimizerInfo,
                                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                                  &sliceNumber,
                                                                                  sizeof(sliceNumber)));
                
                if(flopCount > flopsToStartMPI_)
                {
                    mpirun = true;

                    HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
                    HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));

                    SendSignalToMPICommWorld(0);

                    SendContractionMetaDataToMPICommWorld(modesIn,
                                                          extentsIn,
                                                          qualifiersIn,
                                                          tensorsIn,
                                                          modesOut, 
                                                          dimOut,
                                                          workSpacePreference,
                                                          optimizerAttributes,
                                                          0);

                    HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm));
                    HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle, &cutnComm, sizeof(cutnComm)));

                    HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

                    HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));
                    SetContractionOptimizerAttributes(handle, optimizerConfig, optimizerAttributes);

                    HANDLE_CUTN_ERROR(cutensornetContractionOptimize(handle, 
                                                                     descNet, 
                                                                     optimizerConfig, 
                                                                     limit, 
                                                                     optimizerInfo));
                }
            }

            cutensornetWorkspaceDescriptor_t workDesc;
            HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

            HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeContractionSizes(handle, 
                                                                          descNet, 
                                                                          optimizerInfo, 
                                                                          workDesc));
            
            int64_t worksize_scratch = 0;
            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(handle, 
                                                                workDesc,
                                                                workSpacePreference, 
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                &worksize_scratch));

            if(worksize_scratch > limit)
            {
                throw std::runtime_error("ContractTensors: "
                                         "The required size of scratch exceeds the current workspace limit.");
            }

            void *scratch_ptr = nullptr;

            if(worksize_scratch > 0) 
            {
                HANDLE_CUDA_ERROR(cudaMalloc(&scratch_ptr, worksize_scratch));

                HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, 
                                                                workDesc,
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                scratch_ptr, 
                                                                worksize_scratch));
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkPrepareContraction(handle,
                                                                   descNet,
                                                                   workDesc));

            for(size_t t = 0; t < numInputTensors; ++t)
            {
                HANDLE_CUTN_ERROR(cutensornetNetworkSetInputTensorMemory(handle,
                                                                         descNet,
                                                                         tensorIDs[t],
                                                                         tensorsIn[t],
                                                                         nullptr));
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkSetOutputTensorMemory(handle,
                                                                      descNet,
                                                                      tensorOut,
                                                                      nullptr));

            auto it = std::find_if(optimizerAttributes.begin(), optimizerAttributes.end(), 
                                   [](const auto& pair) {return pair.first == CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING;});

            int32_t disableSlicing = 0;

            if(it != optimizerAttributes.end())
            {
                disableSlicing = it->second;
            }
            
            cutensornetSliceGroup_t sliceGroup = nullptr;

            if(disableSlicing == 0)
            {
                int64_t numSlices = 0;

                HANDLE_CUTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                                  optimizerInfo,
                                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                                  &numSlices,
                                                                                  sizeof(numSlices)));
                
                HANDLE_CUTN_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkContract(handle,
                                                         descNet,
                                                         0,
                                                         workDesc,
                                                         sliceGroup,
                                                         stream));

            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
            
            HANDLE_CUTN_ERROR(cutensornetDestroySliceGroup(sliceGroup));
            HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
            HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
            HANDLE_CUTN_ERROR(cutensornetDestroyNetwork(descNet));
            HANDLE_CUDA_ERROR(cudaFree(scratch_ptr));

            if(pathData.empty())
            {
                HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
            }

            if(mpirun)
            {
                HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle, nullptr, 0));
                HANDLE_MPI_ERROR(MPI_Comm_free(&cutnComm));
            }
        }

        void ApplyTensorSVD(cutensornetHandle_t handle,
                            cudaStream_t stream,
                            cutensornetTensorSVDConfig_t svdConfig,
                            const std::vector<int32_t>& modesInAB,
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
                            const cutensornetWorksizePref_t& workSpacePreference,
                            bool verbose)
        {
            cutensornetTensorDescriptor_t descTensorInAB;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorDescriptor(handle,
                                                                static_cast<int32_t>(modesInAB.size()),
                                                                extentsInAB.data(),
                                                                nullptr, 
                                                                modesInAB.data(),
                                                                typeData_,
                                                                &descTensorInAB));
            
            cutensornetTensorDescriptor_t descTensorOutA;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorDescriptor(handle,
                                                                static_cast<int32_t>(modesOutA.size()),
                                                                extentsOutA.data(),
                                                                nullptr, 
                                                                modesOutA.data(),
                                                                typeData_,
                                                                &descTensorOutA));

            cutensornetTensorDescriptor_t descTensorOutB;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorDescriptor(handle,
                                                                static_cast<int32_t>(modesOutB.size()),
                                                                extentsOutB.data(),
                                                                nullptr, 
                                                                modesOutB.data(),
                                                                typeData_,
                                                                &descTensorOutB));

            cutensornetWorkspaceDescriptor_t workDesc;
            HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

            HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeSVDSizes(handle, 
                                                                  descTensorInAB, 
                                                                  descTensorOutA, 
                                                                  descTensorOutB, 
                                                                  svdConfig,
                                                                  workDesc));

            int64_t device_worksize_scratch = 0;
            int64_t host_worksize_scratch = 0;

            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(handle, 
                                                                workDesc,
                                                                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, 
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                &device_worksize_scratch));

            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(handle, 
                                                                workDesc,
                                                                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, 
                                                                CUTENSORNET_MEMSPACE_HOST, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                &host_worksize_scratch));

            void *device_scratch_ptr = nullptr;
            void *host_scratch_ptr = nullptr;

            if(device_worksize_scratch > 0) 
            {
                HANDLE_CUDA_ERROR(cudaMalloc(&device_scratch_ptr, device_worksize_scratch));

                HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, 
                                                                workDesc,
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                device_scratch_ptr, 
                                                                device_worksize_scratch));
            }

            if(host_worksize_scratch > 0) 
            {
                host_scratch_ptr = malloc(host_worksize_scratch);

                HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, 
                                                                workDesc,
                                                                CUTENSORNET_MEMSPACE_HOST, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                host_scratch_ptr, 
                                                                host_worksize_scratch));
            }

            cutensornetTensorSVDInfo_t svdInfo;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorSVDInfo(handle, &svdInfo));

            HANDLE_CUTN_ERROR(cutensornetTensorSVD(handle, 
                                                   descTensorInAB, tensorInAB, 
                                                   descTensorOutA, tensorOutA,
                                                   nullptr,
                                                   descTensorOutB, tensorOutB, 
                                                   svdConfig, 
                                                   svdInfo,
                                                   workDesc,
                                                   stream));

            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
            
            if(verbose)
            {
                int64_t fullExtent;
                int64_t reducedExtent;
                double discardedWeight;

                HANDLE_CUTN_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, 
                                                                       svdInfo, 
                                                                       CUTENSORNET_TENSOR_SVD_INFO_FULL_EXTENT, 
                                                                       &fullExtent, 
                                                                       sizeof(fullExtent)));
                HANDLE_CUTN_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, 
                                                                       svdInfo, 
                                                                       CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, 
                                                                       &reducedExtent, 
                                                                       sizeof(reducedExtent)));
                HANDLE_CUTN_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, 
                                                                       svdInfo, 
                                                                       CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, 
                                                                       &discardedWeight, 
                                                                       sizeof(discardedWeight)));

                std::cout << "ApplyTensorSVD: Virtual bond truncated from " << fullExtent 
                          << " to " << reducedExtent << std::scientific << std::setprecision(5)
                          << " with a discarded weight " << discardedWeight << std::endl << std::defaultfloat;
            }
        
            int32_t numModesA = static_cast<int32_t>(modesOutA.size());
            std::vector<int64_t> extA(numModesA);
            HANDLE_CUTN_ERROR(cutensornetGetTensorDetails(handle, descTensorOutA, &numModesA, nullptr, nullptr, extA.data(), nullptr));

            newABbondExtent = extA.at(bondIndices.first);

            HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorSVDInfo(svdInfo));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorDescriptor(descTensorInAB));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorDescriptor(descTensorOutA));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorDescriptor(descTensorOutB));
            HANDLE_CUDA_ERROR(cudaFree(device_scratch_ptr));
            free(host_scratch_ptr);
        }

        void ApplyTensorQR(cutensornetHandle_t handle,
                           cudaStream_t stream,
                           const std::vector<int32_t>& modesInAB,
                           const std::vector<int64_t>& extentsInAB,
                           const std::vector<int32_t>& modesOutA,
                           const std::vector<int64_t>& extentsOutA,
                           const std::vector<int32_t>& modesOutB,
                           const std::vector<int64_t>& extentsOutB,
                           const void* tensorInAB,
                           void* tensorOutA,
                           void* tensorOutB)
        {
            cutensornetTensorDescriptor_t descTensorInAB;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorDescriptor(handle,
                                                                static_cast<int32_t>(modesInAB.size()),
                                                                extentsInAB.data(),
                                                                nullptr, 
                                                                modesInAB.data(),
                                                                typeData_,
                                                                &descTensorInAB));
            
            cutensornetTensorDescriptor_t descTensorOutA;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorDescriptor(handle,
                                                                static_cast<int32_t>(modesOutA.size()),
                                                                extentsOutA.data(),
                                                                nullptr, 
                                                                modesOutA.data(),
                                                                typeData_,
                                                                &descTensorOutA));

            cutensornetTensorDescriptor_t descTensorOutB;
            HANDLE_CUTN_ERROR(cutensornetCreateTensorDescriptor(handle,
                                                                static_cast<int32_t>(modesOutB.size()),
                                                                extentsOutB.data(),
                                                                nullptr, 
                                                                modesOutB.data(),
                                                                typeData_,
                                                                &descTensorOutB));

            cutensornetWorkspaceDescriptor_t workDesc;
            HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

            HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeQRSizes(handle, 
                                                                  descTensorInAB, 
                                                                  descTensorOutA, 
                                                                  descTensorOutB, 
                                                                  workDesc));

            int64_t device_worksize_scratch = 0;
            int64_t host_worksize_scratch = 0;

            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(handle, 
                                                                workDesc,
                                                                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, 
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                &device_worksize_scratch));

            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(handle, 
                                                                workDesc,
                                                                CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, 
                                                                CUTENSORNET_MEMSPACE_HOST, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                &host_worksize_scratch));

            void *device_scratch_ptr = nullptr;
            void *host_scratch_ptr = nullptr;

            if(device_worksize_scratch > 0) 
            {
                HANDLE_CUDA_ERROR(cudaMalloc(&device_scratch_ptr, device_worksize_scratch));

                HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, 
                                                                workDesc,
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                device_scratch_ptr, 
                                                                device_worksize_scratch));
            }

            if(host_worksize_scratch > 0) 
            {
                host_scratch_ptr = malloc(host_worksize_scratch);

                HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, 
                                                                workDesc,
                                                                CUTENSORNET_MEMSPACE_HOST, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                host_scratch_ptr, 
                                                                host_worksize_scratch));
            }

            HANDLE_CUTN_ERROR(cutensornetTensorQR(handle, 
                                                  descTensorInAB, tensorInAB, 
                                                  descTensorOutA, tensorOutA,
                                                  descTensorOutB, tensorOutB, 
                                                  workDesc,
                                                  stream));

            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

            HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorDescriptor(descTensorInAB));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorDescriptor(descTensorOutA));
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorDescriptor(descTensorOutB));
            HANDLE_CUDA_ERROR(cudaFree(device_scratch_ptr));
            free(host_scratch_ptr);
        }

        void SendContractionMetaDataToMPICommWorld(const std::vector<std::vector<int32_t>>& modesIn,
                                                   const std::vector<std::vector<int64_t>>& extentsIn,
                                                   const std::vector<cutensornetTensorQualifiers_t>& qualifiersIn,
                                                   const std::vector<const void*>& tensorsIn,
                                                   const std::vector<int32_t>& modesOut,
                                                   size_t dimOut,
                                                   const cutensornetWorksizePref_t& workSpacePreference,
                                                   const ContractionOptimizerAttributes& optimizerAttributes,
                                                   int32_t numAutotuningIterations)
        {
            if(!MPI_) return;

            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            if(rank != 0)
            {
                throw std::runtime_error("SendContractionMetaDataToMPICommWorld: "
                                         "The method can be only executed by the main process (rank == 0).");
            }

            size_t numInputTensors = modesIn.size();
            MPI_Bcast((void*)(&numInputTensors), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            for (size_t i = 0; i < numInputTensors; ++i) 
            {
                size_t numModes = modesIn[i].size();
                MPI_Bcast((void*)(&numModes), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

                MPI_Bcast((void*)(modesIn[i].data()), numModes, MPI_INT32_T, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(extentsIn[i].data()), numModes, MPI_INT64_T, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(&qualifiersIn[i]), sizeof(cutensornetTensorQualifiers_t), MPI_BYTE, 0, MPI_COMM_WORLD);

                bool isNullPtr = tensorsIn[i] == nullptr;
                MPI_Bcast((void*)(&isNullPtr), 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
                
                if(!isNullPtr)
                {
                    int64_t dim = std::accumulate(extentsIn[i].begin(),
                                                  extentsIn[i].end(),
                                                  1L,
                                                  [](int64_t acc, const int64_t& current) {return acc * current;});
                    
                    MPI_Bcast(const_cast<void*>(tensorsIn[i]), dim * sizeof(complexType), MPI_BYTE, 0, MPI_COMM_WORLD);
                }
            }

            size_t numModesOut = modesOut.size();
            MPI_Bcast((void*)(&numModesOut), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            if(numModesOut > 0UL)
            {
                MPI_Bcast((void*)(modesOut.data()), numModesOut, MPI_INT32_T, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(&dimOut), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            }

            MPI_Bcast((void*)(&workSpacePreference), sizeof(cutensornetWorksizePref_t), MPI_BYTE, 0, MPI_COMM_WORLD);

            size_t numAttributes = optimizerAttributes.size();
            MPI_Bcast((void*)(&numAttributes), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            for(auto& [attr, val] : optimizerAttributes)
            {
                MPI_Bcast((void*)(&attr), sizeof(cutensornetContractionOptimizerConfigAttributes_t), MPI_BYTE, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(&val), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
            }

            MPI_Bcast((void*)(&numAutotuningIterations), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        }

        void ReceiveContractionMetaDataFromMPICommWorld(std::vector<std::vector<int32_t>>& modesIn,
                                                        std::vector<std::vector<int64_t>>& extentsIn,
                                                        std::vector<cutensornetTensorQualifiers_t>& qualifiersIn,
                                                        std::vector<void*>& tensorsIn,
                                                        std::vector<int32_t>& modesOut,
                                                        size_t& dimOut,
                                                        cutensornetWorksizePref_t& workSpacePreference,
                                                        ContractionOptimizerAttributes& optimizerAttributes,
                                                        int32_t& numAutotuningIterations)
        {       
            if(!MPI_) return;

            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            if(rank == 0)
            {
                throw std::runtime_error("ReceiveContractionMetaDataFromMPICommWorld: "
                                         "The method must be executed by the secondary processes (rank != 0).");
            }

            size_t numInputTensors;
            MPI_Bcast((void*)(&numInputTensors), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            modesIn.resize(numInputTensors);
            extentsIn.resize(numInputTensors);
            qualifiersIn.resize(numInputTensors);
            tensorsIn.resize(numInputTensors);

            for (size_t i = 0; i < numInputTensors; ++i) 
            {
                size_t numModes;
                MPI_Bcast((void*)(&numModes), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

                modesIn[i].resize(numModes);
                extentsIn[i].resize(numModes);

                MPI_Bcast((void*)(modesIn[i].data()), numModes, MPI_INT32_T, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(extentsIn[i].data()), numModes, MPI_INT64_T, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(&qualifiersIn[i]), sizeof(cutensornetTensorQualifiers_t), MPI_BYTE, 0, MPI_COMM_WORLD);

                bool isNullPtr;
                MPI_Bcast((void*)(&isNullPtr), 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
                
                if(!isNullPtr)
                {
                    int64_t dim = std::accumulate(extentsIn[i].begin(),
                                                  extentsIn[i].end(),
                                                  1L,
                                                  [](int64_t acc, const int64_t& current) {return acc * current;});
                    
                    HANDLE_CUDA_ERROR(cudaMalloc(&tensorsIn[i], static_cast<size_t>(dim) * sizeof(complexType)));

                    MPI_Bcast(tensorsIn[i], dim * sizeof(complexType), MPI_BYTE, 0, MPI_COMM_WORLD);
                }
            }

            size_t numModesOut;
            MPI_Bcast((void*)(&numModesOut), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            modesOut.resize(numModesOut);

            if(numModesOut > 0UL)
            {
                MPI_Bcast((void*)(modesOut.data()), numModesOut, MPI_INT32_T, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(&dimOut), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            }
            else
            {
                dimOut = 1UL;
            }

            MPI_Bcast((void*)(&workSpacePreference), sizeof(cutensornetWorksizePref_t), MPI_BYTE, 0, MPI_COMM_WORLD);

            size_t numAttributes;
            MPI_Bcast((void*)(&numAttributes), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            optimizerAttributes.resize(numAttributes);

            for(auto& [attr, val] : optimizerAttributes)
            {
                MPI_Bcast((void*)(&attr), sizeof(cutensornetContractionOptimizerConfigAttributes_t), MPI_BYTE, 0, MPI_COMM_WORLD);
                MPI_Bcast((void*)(&val), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
            }

            MPI_Bcast((void*)(&numAutotuningIterations), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        }

        void SendSignalToMPICommWorld(int32_t key) 
        {       
            if(!MPI_) return;

            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            if(rank != 0)
            {
                throw std::runtime_error("SendSignalToMPICommWorld: "
                                         "The method can be only executed by the main process (rank == 0).");
            }

            MPI_Bcast((void*)(&key), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        }

        void MPIContractionHelper()
        {
            if(!MPI_) return;

            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            if(rank == 0)
            {
                throw std::runtime_error("MPIContractionHelper: "
                                         "The method must be executed by the secondary processes (rank != 0).");
            }

            while(true)
            {
                int key;
                MPI_Bcast((void*)(&key), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

                if((key != 0) && (key != 1)) break;
                
                cutensornetHandle_t handle;
                HANDLE_CUTN_ERROR(cutensornetCreate(&handle));

                cudaStream_t stream;
                HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

                std::vector<std::vector<int32_t>> modesIn;
                std::vector<std::vector<int64_t>> extentsIn;
                std::vector<cutensornetTensorQualifiers_t> qualifiersIn;
                std::vector<void*> tensorsIn;
                std::vector<int32_t> modesOut;
                size_t dimOut;
                cutensornetWorksizePref_t workSpacePreference;
                ContractionOptimizerAttributes optimizerAttributes;
                int32_t numAutotuningIterations;

                ReceiveContractionMetaDataFromMPICommWorld(modesIn,
                                                           extentsIn,
                                                           qualifiersIn,
                                                           tensorsIn,
                                                           modesOut,
                                                           dimOut,
                                                           workSpacePreference,
                                                           optimizerAttributes,
                                                           numAutotuningIterations);

                std::vector<const void*> const_tensorsIn(tensorsIn.begin(), tensorsIn.end());

                void* tensorOut;
                HANDLE_CUDA_ERROR(cudaMalloc(&tensorOut, dimOut * sizeof(complexType)));

                if(key == 0)
                {                    
                    ContractTensors(handle,
                                    stream,
                                    modesIn,
                                    extentsIn,
                                    qualifiersIn,
                                    const_tensorsIn,
                                    modesOut,
                                    dimOut,
                                    tensorOut,
                                    0.8,
                                    0UL,
                                    {},
                                    workSpacePreference,
                                    optimizerAttributes,
                                    true);
                }
                else if(key == 1)
                {                                                                               
                    TensorNetDescriptor opTNDesc(modesIn, 
                                                 extentsIn, 
                                                 qualifiersIn, 
                                                 modesOut, 
                                                 dimOut);
                    
                    OperatorVectorProduct* opMatVecProduct = BuildOperatorVectorProduct(handle,
                                                                                        stream,
                                                                                        opTNDesc,
                                                                                        const_tensorsIn,
                                                                                        0.8,
                                                                                        0UL,
                                                                                        workSpacePreference,
                                                                                        optimizerAttributes,
                                                                                        numAutotuningIterations,
                                                                                        true);

                    void* tensorIn;
                    HANDLE_CUDA_ERROR(cudaMalloc(&tensorIn, opMatVecProduct->GetInTensorSize() * sizeof(complexType)));

                    while(true)
                    {
                        bool start;
                        MPI_Bcast((void*)(&start), 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

                        if(!start) break;
                        
                        opMatVecProduct->GetProductFunction()(tensorIn, tensorOut);
                    }

                    HANDLE_CUDA_ERROR(cudaFree(tensorIn));
                                                                                    
                    opMatVecProduct->ClearTNMetaData();

                    delete opMatVecProduct;
                }

                for(auto ten : tensorsIn)
                {
                    HANDLE_CUDA_ERROR(cudaFree(ten));
                }

                HANDLE_CUDA_ERROR(cudaFree(tensorOut));

                HANDLE_CUTN_ERROR(cutensornetDestroy(handle));
                HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
            }
        }

        OperatorVectorProduct* BuildOperatorVectorProduct(cutensornetHandle_t handle,
                                                          cudaStream_t stream,
                                                          const TensorNetDescriptor& rOmegaTNDesc,
                                                          const std::vector<const void*>& rOmegaInTensors,
                                                          double memoryFactor,
                                                          uint64_t workSpaceLimit,
                                                          const cutensornetWorksizePref_t& workSpacePreference,
                                                          const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes,
                                                          int32_t numAutotuningIterations,
                                                          bool mpi)
        {
            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
            
            const auto& rOmegaInModes = rOmegaTNDesc.GetInModes();
            const auto& rOmegaInExtents = rOmegaTNDesc.GetInExtents();
            const auto& rOmegaInQualifiers = rOmegaTNDesc.GetInQualifiers();
            const auto& rOmegaOutModes = rOmegaTNDesc.GetOutModes();
            
            size_t dimOut = rOmegaTNDesc.GetOutTensorSize();
            size_t numInputTensors = rOmegaInModes.size();
            
            cutensornetNetworkDescriptor_t descNet;
            HANDLE_CUTN_ERROR(cutensornetCreateNetwork(handle, &descNet));

            std::vector<int64_t> tensorIDs(numInputTensors);

            for(size_t t = 0; t < numInputTensors; ++t)
            {
                HANDLE_CUTN_ERROR(cutensornetNetworkAppendTensor(handle,
                                                                 descNet,
                                                                 static_cast<int32_t>(rOmegaInModes[t].size()),
                                                                 rOmegaInExtents[t].data(),
                                                                 rOmegaInModes[t].data(),
                                                                 &rOmegaInQualifiers[t],
                                                                 CuTensorNetMethods::typeData_,
                                                                 &tensorIDs[t]));
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkSetOutputTensor(handle,
                                                                descNet,
                                                                static_cast<int32_t>(rOmegaOutModes.size()),
                                                                rOmegaOutModes.data(),
                                                                CuTensorNetMethods::typeData_));
            
            
            HANDLE_CUTN_ERROR(cutensornetNetworkSetAttribute(handle,
                                                             descNet,
                                                             CUTENSORNET_NETWORK_COMPUTE_TYPE,
                                                             &CuTensorNetMethods::typeCompute_,
                                                             sizeof(CuTensorNetMethods::typeCompute_)));

            size_t freeMem, totalMem;
            HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
            uint64_t limit = static_cast<uint64_t>(static_cast<double>(freeMem) * memoryFactor);

            if((limit > workSpaceLimit) && (workSpaceLimit > 0UL))
            {
                limit = workSpaceLimit;
            }

            bool mpirun = false;
            MPI_Comm cutnComm = MPI_COMM_NULL;

            if((rank > 0) && mpi && MPI_)
            {
                mpirun = true;

                HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm));
                HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle, &cutnComm, sizeof(cutnComm)));
            }

            cutensornetContractionOptimizerInfo_t optimizerInfo;
            cutensornetContractionOptimizerConfig_t optimizerConfig;

            HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

            HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));
            SetContractionOptimizerAttributes(handle, optimizerConfig, optimizerAttributes);

            HANDLE_CUTN_ERROR(cutensornetContractionOptimize(handle, 
                                                             descNet, 
                                                             optimizerConfig, 
                                                             limit, 
                                                             optimizerInfo));

            if((rank == 0) && mpi && MPI_)
            {         
                double flopCount;
                HANDLE_CUTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                                  optimizerInfo,
                                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                                                                                  &flopCount,
                                                                                  sizeof(flopCount)));
                
                if(flopCount > flopsToStartMPI_)
                {
                    mpirun = true;

                    HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
                    HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));

                    SendSignalToMPICommWorld(1);

                    SendContractionMetaDataToMPICommWorld(rOmegaInModes,
                                                          rOmegaInExtents,
                                                          rOmegaInQualifiers,
                                                          rOmegaInTensors,
                                                          rOmegaOutModes, 
                                                          dimOut,
                                                          workSpacePreference,
                                                          optimizerAttributes,
                                                          numAutotuningIterations);

                    HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm));
                    HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle, &cutnComm, sizeof(cutnComm)));

                    HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

                    HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));
                    SetContractionOptimizerAttributes(handle, optimizerConfig, optimizerAttributes);

                    HANDLE_CUTN_ERROR(cutensornetContractionOptimize(handle, 
                                                                     descNet, 
                                                                     optimizerConfig, 
                                                                     limit, 
                                                                     optimizerInfo));
                }
            }
            
            cutensornetWorkspaceDescriptor_t workDesc;
            HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
            
            HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeContractionSizes(handle, 
                                                                          descNet, 
                                                                          optimizerInfo, 
                                                                          workDesc));
            
            int64_t worksize_scratch = 0;
            
            HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(handle, 
                                                                workDesc,
                                                                workSpacePreference, 
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                &worksize_scratch));
            
            if(worksize_scratch > limit)
            {
                throw std::runtime_error("TensorNetwork::BuildOperatorVectorProduct: "
                                         "The required size of scratch exceeds the current workspace limit.");
            }

            void *scratch_ptr = nullptr;

            if(worksize_scratch > 0) 
            {
                HANDLE_CUDA_ERROR(cudaMalloc(&scratch_ptr, worksize_scratch));
            
                HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(handle, 
                                                                workDesc,
                                                                CUTENSORNET_MEMSPACE_DEVICE, 
                                                                CUTENSORNET_WORKSPACE_SCRATCH, 
                                                                scratch_ptr, 
                                                                worksize_scratch));
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkPrepareContraction(handle,
                                                                   descNet,
                                                                   workDesc));

            for(size_t t = 0; t < numInputTensors - 1UL; ++t)
            {
                HANDLE_CUTN_ERROR(cutensornetNetworkSetInputTensorMemory(handle,
                                                                         descNet,
                                                                         tensorIDs[t],
                                                                         rOmegaInTensors[t],
                                                                         nullptr));
            }

            cutensornetNetworkAutotunePreference_t autotunePref;
            HANDLE_CUTN_ERROR(cutensornetCreateNetworkAutotunePreference(handle, &autotunePref));

            HANDLE_CUTN_ERROR(cutensornetNetworkAutotunePreferenceSetAttribute(handle,
                                                                               autotunePref,
                                                                               CUTENSORNET_NETWORK_AUTOTUNE_MAX_ITERATIONS,
                                                                               &numAutotuningIterations,
                                                                               sizeof(numAutotuningIterations)));

            auto it = std::find_if(optimizerAttributes.begin(), optimizerAttributes.end(), 
                                   [](const auto& pair) {return pair.first == CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING;});

            int32_t disableSlicing = 0;

            if(it != optimizerAttributes.end())
            {
                disableSlicing = it->second;
            }

            cutensornetSliceGroup_t sliceGroup = nullptr;
            
            if(disableSlicing == 0)
            {
                int64_t numSlices = 0;

                HANDLE_CUTN_ERROR(cutensornetContractionOptimizerInfoGetAttribute(handle,
                                                                                  optimizerInfo,
                                                                                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                                                                                  &numSlices,
                                                                                  sizeof(numSlices)));
                
                HANDLE_CUTN_ERROR(cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup));
            }

            size_t dimIn = static_cast<size_t>(std::accumulate(rOmegaInExtents.back().begin(),
                                                               rOmegaInExtents.back().end(),
                                                               1L,
                                                               [](int64_t acc, const int64_t& current) {return acc * current;}));

            return new OperatorVectorProduct(handle, 
                                             stream,
                                             cutnComm, 
                                             descNet, 
                                             tensorIDs.back(), 
                                             optimizerConfig,
                                             optimizerInfo,
                                             workDesc, 
                                             scratch_ptr, 
                                             autotunePref, 
                                             sliceGroup,
                                             dimIn,
                                             dimOut,
                                             mpirun);
        }
    }
}