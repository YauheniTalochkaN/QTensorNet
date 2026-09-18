#include "TensorDescriptor.hh"

namespace QTensorNet
{
    TensorDescriptor::TensorDescriptor(const std::vector<int32_t>& modes, 
                                       const std::vector<int64_t>& extents) : 
                                       modes_(modes), extents_(extents)
    {

    }

    TensorDescriptor::~TensorDescriptor()
    {
        
    }

    size_t TensorDescriptor::GetTensorSize() const
    {
        return static_cast<size_t>(std::accumulate(extents_.begin(),
                                                   extents_.end(),
                                                   1L,
                                                   [](int64_t acc, const int64_t& current) {return acc * current;}));
    }

    const std::vector<int32_t>& TensorDescriptor::GetModes() const
    {
        return modes_;
    }

    const std::vector<int64_t>& TensorDescriptor::GetExtents() const
    {
        return extents_;
    }
}