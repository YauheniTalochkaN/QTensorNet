#pragma once

#include <vector>
#include <complex>

#include <cuda_runtime.h>

#include "CuErrorUtils.hh"

namespace QTensorNet
{
    namespace CuArrayMethods
    {
        using complexType = std::complex<double>;

        void* VectorToGPUArray(const std::vector<complexType>& data_host);
        std::vector<complexType> GPUArrayToVector(const void* data_device, size_t size);
    }
}