#include "CachedLeaves.hh"

namespace QTensorNet
{
    CachedLeaves::CachedLeaves()
    {

    }

    CachedLeaves::~CachedLeaves()
    {
        
    }

    void CachedLeaves::Clear()
    {
        ltensors_.clear();
        lsites_.clear();
        lmodes_.clear();
        lextents_.clear();
    }

    void CachedLeaves::AddLeave(const std::set<size_t>& sites,  
                                const std::vector<int32_t>& modes, 
                                const std::vector<int64_t>& extents,
                                void* data)
    {
        auto it = std::find(lsites_.begin(), lsites_.end(), sites);

        auto deleter = [](void* ptr)
        {
            if(ptr != nullptr)
            {
                HANDLE_CUDA_ERROR(cudaFree(ptr));
            }
        };

        if(it == lsites_.end())
        {
            lsites_.push_back(sites);
            lmodes_.push_back(modes);
            lextents_.push_back(extents);
            ltensors_.emplace_back(data, deleter);
        }
        else
        {
            size_t i = std::distance(lsites_.begin(), it);
            
            lmodes_[i] = modes;
            lextents_[i] = extents;

            if (data != ltensors_[i].get()) 
            {
                ltensors_[i] = std::shared_ptr<void>(data, deleter);
            }
        }
    }
    
    void CachedLeaves::EraseLeavesIf(const std::set<size_t>& sites)
    {
        for(size_t i = 0UL; i < lsites_.size();)
        {
            const auto& s = lsites_[i];
            
            bool is = std::any_of(sites.begin(), sites.end(), 
                                  [s](size_t val) {return s.find(val) != s.end();});

            if(is)
            {
                lsites_.erase(lsites_.begin() + i);
                lmodes_.erase(lmodes_.begin() + i);
                lextents_.erase(lextents_.begin() + i);
                ltensors_.erase(ltensors_.begin() + i);
            }
            else
            {
                ++i;
            }
        }
    }

    std::pair<std::shared_ptr<void>, TensorDescriptor> CachedLeaves::GetLeave(size_t id)
    {
        if(id >= lsites_.size())
        {
            throw std::invalid_argument("CachedLeaves::GetLeave: "
                                        "Leave index exceeds the maximum number.");
        }

        return {ltensors_[id], TensorDescriptor(lmodes_[id], lextents_[id])};
    }

    size_t CachedLeaves::FindSubLeaveIndex(const std::set<size_t>& leave)
    {
        size_t bestIdx = std::numeric_limits<size_t>::max();
        size_t bestSize = 0UL;

        for(size_t i = 0; i < lsites_.size(); ++i) 
        {
            const auto& candidate = lsites_[i];

            if(std::includes(leave.begin(), leave.end(), candidate.begin(), candidate.end()))
            {
                if(candidate.size() > bestSize) 
                {
                    bestSize = candidate.size();
                    bestIdx = i;
                }
            }
        }

        return bestIdx;
    }

    const void* CachedLeaves::GetTensorData(size_t id) const
    {
        if(id >= ltensors_.size())
        {
            throw std::invalid_argument("CachedLeaves::GetTensorData: "
                                        "Tensor index exceeds the maximum number.");
        }
        
        return ltensors_[id].get();
    }
    
    const std::set<size_t>& CachedLeaves::GetSites(size_t id) const
    {
        if(id >= lsites_.size())
        {
            throw std::invalid_argument("CachedLeaves::GetSites: "
                                        "The index of site set exceeds the maximum number.");
        }
        
        return lsites_[id];
    }

    const std::vector<int32_t>& CachedLeaves::GetModes(size_t id) const
    {
        if(id >= lmodes_.size())
        {
            throw std::invalid_argument("CachedLeaves::GetModes: "
                                        "The index of mode vector exceeds the maximum number.");
        }
        
        return lmodes_[id];
    }

    const std::vector<int64_t>& CachedLeaves::GetExtents(size_t id) const
    {
        if(id >= lextents_.size())
        {
            throw std::invalid_argument("CachedLeaves::GetExtents: "
                                        "The index of extent vector exceeds the maximum number.");
        }
        
        return lextents_[id];
    }
}