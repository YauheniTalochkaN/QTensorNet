#include "TensorNetDescriptor.hh"

namespace QTensorNet
{
    TensorNetDescriptor::TensorNetDescriptor(const std::vector<std::vector<int32_t>>& inModes,
                                             const std::vector<std::vector<int64_t>>& inExtents,
                                             const std::vector<cutensornetTensorQualifiers_t>& inQualifiers,
                                             const std::vector<int32_t>& outModes,
                                             size_t outDim) : 
                                             inModes_(inModes),
                                             inExtents_(inExtents),
                                             inQualifiers_(inQualifiers),
                                             outModes_(outModes),
                                             outDim_(outDim)
    {        
        
    }

    TensorNetDescriptor::~TensorNetDescriptor()
    {

    }

    size_t TensorNetDescriptor::GetOutTensorSize() const
    {
        return outDim_;
    }

    const std::vector<std::vector<int32_t>>& TensorNetDescriptor::GetInModes() const
    {
        return inModes_;
    }

    const std::vector<std::vector<int64_t>>& TensorNetDescriptor::GetInExtents() const
    {
        return inExtents_;
    }

    const std::vector<cutensornetTensorQualifiers_t>& TensorNetDescriptor::GetInQualifiers() const
    {
        return inQualifiers_;
    }

    const std::vector<int32_t>& TensorNetDescriptor::GetOutModes() const
    {
        return outModes_;
    }
}