#include "CuArrayMethods.hh"

namespace QTensorNet
{
    namespace CuArrayMethods
    {    
        void* VectorToGPUArray(const std::vector<complexType>& data_host)
        {
            void* data_device;

            HANDLE_CUDA_ERROR(cudaMalloc(&data_device, data_host.size() * sizeof(complexType)));
            HANDLE_CUDA_ERROR(cudaMemcpy(data_device, data_host.data(), data_host.size() * sizeof(complexType), cudaMemcpyHostToDevice));

            return data_device;
        }

        std::vector<complexType> GPUArrayToVector(const void* data_device, size_t size)
        {
            std::vector<complexType> data_host(size);

            HANDLE_CUDA_ERROR(cudaMemcpy(data_host.data(), data_device, size * sizeof(complexType), cudaMemcpyDeviceToHost));

            return data_host;
        }
    }
}