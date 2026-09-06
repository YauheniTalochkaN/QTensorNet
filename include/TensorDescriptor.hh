#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <numeric>

namespace QTensorNet
{
    class TensorDescriptor
    {
    public:
        TensorDescriptor(const std::vector<int32_t>& modes, 
                         const std::vector<int64_t>& extents);
        ~TensorDescriptor();
        TensorDescriptor(const TensorDescriptor&) = default;
        TensorDescriptor(TensorDescriptor&&) = default;
        TensorDescriptor& operator=(const TensorDescriptor&) = default;
        TensorDescriptor& operator=(TensorDescriptor&&) = default;
        size_t GetTensorSize() const;
        const std::vector<int32_t>& GetModes() const;
        const std::vector<int64_t>& GetExtents() const;

    private:
        std::vector<int32_t> modes_;
        std::vector<int64_t> extents_;
    };
}