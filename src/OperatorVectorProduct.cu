#include "OperatorVectorProduct.hh"

using complexType = std::complex<double>;

namespace QTensorNet
{
    OperatorVectorProduct::OperatorVectorProduct(cutensornetHandle_t handle,
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
                                                 bool mpirun) :
                                                 handle_(handle), 
                                                 stream_(stream),
                                                 cutnComm_(cutnComm), 
                                                 descNet_(descNet),
                                                 lastTensorID_(lastTensorID),
                                                 optimizerConfig_(optimizerConfig),
                                                 optimizerInfo_(optimizerInfo),
                                                 workDesc_(workDesc),
                                                 scratchPtr_(scratchPtr),
                                                 autotunePref_(autotunePref),
                                                 sliceGroup_(sliceGroup),
                                                 inTensorSize_(inTensorSize),
                                                 outTensorSize_(outTensorSize),
                                                 mpirun_(mpirun)
    {
        productFunction_ = [this](void* tensorIn, void* tensorOut)
        {
            if(mpirun_)
            {
                int rank{-1};
                HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

                if(rank == 0)
                {                    
                    bool start = true;
                    MPI_Bcast((void*)(&start), 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
                }
                
                MPI_Bcast(tensorIn, inTensorSize_ * sizeof(complexType), MPI_BYTE, 0, MPI_COMM_WORLD);
            }

            HANDLE_CUTN_ERROR(cutensornetNetworkSetInputTensorMemory(handle_,
                                                                     descNet_,
                                                                     lastTensorID_,
                                                                     tensorIn,
                                                                     nullptr));
            
            HANDLE_CUTN_ERROR(cutensornetNetworkSetOutputTensorMemory(handle_,
                                                                      descNet_,
                                                                      tensorOut,
                                                                      nullptr));
            if(firstUsage_)
            {
                HANDLE_CUTN_ERROR(cutensornetNetworkAutotuneContraction(handle_,
                                                                        descNet_,
                                                                        workDesc_,
                                                                        autotunePref_,
                                                                        stream_));
                
                firstUsage_ = false;
            }
            
            HANDLE_CUTN_ERROR(cutensornetNetworkContract(handle_,
                                                         descNet_,
                                                         0,
                                                         workDesc_,
                                                         sliceGroup_,
                                                         stream_));
            
            HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream_));
        };
    }

    OperatorVectorProduct::~OperatorVectorProduct()
    {        
        if(mpirun_)
        {
            int rank{-1};
            HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            if(rank == 0)
            {
                bool start = false;
                MPI_Bcast((void*)(&start), 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
            }

            HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle_, nullptr, 0));
            HANDLE_MPI_ERROR(MPI_Comm_free(&cutnComm_));
        }
    }
    
    void OperatorVectorProduct::DestroyDescNet()
    {
        HANDLE_CUTN_ERROR(cutensornetDestroyNetwork(descNet_));
    }
    
    void OperatorVectorProduct::DestroyOptimizerConfig()
    {
        HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig_));
    }
    
    void OperatorVectorProduct::DestroyOptimizerInfo()
    {
        HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo_));
    }
    
    void OperatorVectorProduct::DestroyWorkDesc()
    {
        HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc_));
    }

    void OperatorVectorProduct::DestroyScratchPtr()
    {
        HANDLE_CUDA_ERROR(cudaFree(scratchPtr_));
    }
    
    void OperatorVectorProduct::DestroyAutotunePref()
    {
        HANDLE_CUTN_ERROR(cutensornetDestroyNetworkAutotunePreference(autotunePref_));
    }
    
    void OperatorVectorProduct::DestroySliceGroup()
    {
        HANDLE_CUTN_ERROR(cutensornetDestroySliceGroup(sliceGroup_));
    }

    void OperatorVectorProduct::ClearTNMetaData()
    {
        DestroyDescNet();
        DestroyOptimizerConfig();
        DestroyOptimizerInfo();
        DestroyWorkDesc();
        DestroyScratchPtr();
        DestroyAutotunePref();
        DestroySliceGroup();
    }

    const std::function<void(void*, void*)>& OperatorVectorProduct::GetProductFunction() const
    {
        return productFunction_;
    }

    size_t OperatorVectorProduct::GetInTensorSize() const
    {
        return inTensorSize_;
    }
    
    size_t OperatorVectorProduct::GetOutTensorSize() const
    {
        return outTensorSize_;
    }
}