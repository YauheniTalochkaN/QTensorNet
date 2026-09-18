#pragma once

#include <set>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <utility>
#include <stdexcept>

#include "CuErrorUtils.hh"
#include "TensorDescriptor.hh"

namespace QTensorNet
{
    class CachedLeaves
    {
    public:
        CachedLeaves();
        ~CachedLeaves();
        CachedLeaves(const CachedLeaves&) = default;
        CachedLeaves(CachedLeaves&&) = default;
        CachedLeaves& operator=(const CachedLeaves&) = default;
        CachedLeaves& operator=(CachedLeaves&&) = default;
        void Clear();
        void AddLeave(const std::set<size_t>& sites,  
                      const std::vector<int32_t>& modes, 
                      const std::vector<int64_t>& extents,
                      void* data);
        void EraseLeavesIf(const std::set<size_t>& sites);
        std::pair<std::shared_ptr<void>, TensorDescriptor> GetLeave(size_t id);
        size_t FindSubLeaveIndex(const std::set<size_t>& leave);
        const void* GetTensorData(size_t id) const;
        const std::set<size_t>& GetSites(size_t id) const;
        const std::vector<int32_t>& GetModes(size_t id) const;
        const std::vector<int64_t>& GetExtents(size_t id) const;
        
    private:
        std::vector<std::shared_ptr<void>> ltensors_;
        std::vector<std::set<size_t>> lsites_;
        std::vector<std::vector<int32_t>> lmodes_;
        std::vector<std::vector<int64_t>> lextents_;
    };
}