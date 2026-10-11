#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

#include "CuErrorUtils.hh"

namespace QTensorNet
{
    class TensorNetDescriptor
    {
    public:
        TensorNetDescriptor(const std::vector<std::vector<int32_t>>& inModes,
                            const std::vector<std::vector<int64_t>>& inExtents,
                            const std::vector<cutensornetTensorQualifiers_t>& inQualifiers,
                            const std::vector<int32_t>& outModes,
                            size_t outDim);
        ~TensorNetDescriptor();
        TensorNetDescriptor(const TensorNetDescriptor&) = default;
        TensorNetDescriptor(TensorNetDescriptor&&) = default;
        TensorNetDescriptor& operator=(const TensorNetDescriptor&) = default;
        TensorNetDescriptor& operator=(TensorNetDescriptor&&) = default;
        size_t GetOutTensorSize() const;
        const std::vector<std::vector<int32_t>>& GetInModes() const;
        const std::vector<std::vector<int64_t>>& GetInExtents() const;
        const std::vector<cutensornetTensorQualifiers_t>& GetInQualifiers() const;
        const std::vector<int32_t>& GetOutModes() const;

    private:
        std::vector<std::vector<int32_t>> inModes_;
        std::vector<std::vector<int64_t>> inExtents_;
        std::vector<cutensornetTensorQualifiers_t> inQualifiers_;
        std::vector<int32_t> outModes_;
        size_t outDim_;
    };
}