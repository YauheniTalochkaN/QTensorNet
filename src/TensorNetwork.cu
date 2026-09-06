#include "TensorNetwork.hh"

namespace QTensorNet
{    
    std::vector<double> GetSuzukiCoeffs(size_t k) 
    {
        if (k <= 1UL) 
        {
            return {1.0};
        }

        std::vector<double> previous = GetSuzukiCoeffs(k - 1UL);

        double pk = 1.0 / (4.0 - std::pow(4.0, 1.0 / (2.0 * static_cast<double>(k) - 1.0)));

        std::vector<double> current;
        current.reserve(previous.size() * 5UL);

        auto append_scaled = [&](double scale) 
        {
            for (double p : previous) 
            {
                current.push_back(p * scale);
            }
        };

        append_scaled(pk);
        append_scaled(pk);
        append_scaled(1.0 - 4.0 * pk);
        append_scaled(pk);
        append_scaled(pk);

        return current;
    }

    graphTraversalType GetGraphTraversalToRoot(const graphType& graph, size_t numSites, size_t root, bool& loop_free)
    {       
        if(root >= numSites) 
        {
            throw std::invalid_argument("GetGraphTraversalToRoot: "
                                        "The root index exceeds number of sites.");
        }
        
        std::map<size_t, std::vector<size_t>> adjMap;
        
        for(const auto& edge : graph) 
        {
            adjMap[edge.first].push_back(edge.second);
            adjMap[edge.second].push_back(edge.first);
        }

        std::vector<bool> visitedNodes(numSites, false);

        graphTraversalType result;

        std::function<bool(size_t, size_t)> dfs = [&](size_t u, size_t parent) 
        {
            bool loop = false;
            
            visitedNodes[u] = true;
            
            for(auto& v : adjMap[u]) 
            {
                if(v == parent) 
                {
                    continue;
                }

                if(!visitedNodes[v]) 
                {
                    bool answer = dfs(v, u);
                    
                    if(answer)
                    {
                        loop = true;
                        break;
                    }
                } 
                else 
                {
                    return true;
                }
            }

            if(parent != std::numeric_limits<size_t>::max()) 
            { 
                result.emplace_back(u, parent);
            }

            return loop;
        };

        std::function<void()> bfs = [&]()
        {
            std::queue<size_t> nodeQueue;
            std::vector<size_t> distance(numSites, std::numeric_limits<size_t>::max());
        
            nodeQueue.push(root);
            distance[root] = 0;
        
            while(!nodeQueue.empty()) 
            {
                size_t currentNode = nodeQueue.front();
                nodeQueue.pop();
            
                for(size_t neighbor : adjMap[currentNode]) 
                {
                    if(distance[neighbor] == std::numeric_limits<size_t>::max()) 
                    {
                        distance[neighbor] = distance[currentNode] + 1UL;
                        result.emplace_back(neighbor, currentNode);
                        nodeQueue.push(neighbor);
                    }
                    else 
                    {
                        if((distance[neighbor] > distance[currentNode]) || 
                           ((distance[neighbor] == distance[currentNode]) && (neighbor > currentNode)))
                        {
                            result.emplace_back(neighbor, currentNode);
                        }
                    }
                }
            }
        
            std::reverse(result.begin(), result.end());
        };

        if(!dfs(root, std::numeric_limits<size_t>::max()))
        {
            loop_free = true;
        }
        else
        {
            loop_free = false;
            
            result.clear();
            visitedNodes.clear();

            bfs();
        }

        return result;
    }

    void BuildOpTensors(virtualModesGraphType& graph, 
                        size_t numSites, 
                        size_t root, 
                        const std::vector<OpTerm>& OpTerms, 
                        std::vector<std::vector<complexType>>& tensors_host)
    {
        graphType local_graph;
        std::vector<std::vector<size_t>> neighbors_(numSites);

        for(const auto& edge : graph) 
        {
            auto [in, out] = std::minmax(std::get<0UL>(edge), std::get<1UL>(edge));

            local_graph.emplace(in, out);

            neighbors_[in].push_back(out);
            neighbors_[out].push_back(in);
        }

        bool loop_free = false;
        graphTraversalType traversal = GetGraphTraversalToRoot(local_graph, numSites, root, loop_free);

        if(!loop_free) 
        {
            throw std::runtime_error("BuildOpTensors:"
                                     "Graph is not loop-free!");
        }

        std::vector<size_t> parent(numSites, std::numeric_limits<size_t>::max());

        for(const auto& edge : traversal) 
        {
            parent[edge.first] = edge.second;
        }

        std::vector<size_t> phys_dims(numSites, 0UL);

        for(const auto& term : OpTerms) 
        {
            for(const auto& op : term) 
            {
                size_t site = std::get<0>(op);

                if(!std::get<2>(op).empty()) 
                {
                    phys_dims[site] = static_cast<size_t>(std::sqrt(std::get<2>(op).size()));
                }
            }
        }

        for(size_t i = 0UL; i < numSites; ++i) 
        {
            if(phys_dims[i] == 0UL)
            {
                throw std::runtime_error("BuildOpTensors:"
                                         "Physical dimension for site " + std::to_string(i) + " is zero.");
            }
        }

        std::vector<size_t> depth(numSites, 0UL);

        for(auto it = traversal.rbegin(); it != traversal.rend(); ++it) 
        {
            depth[it->first] = depth[it->second] + 1UL;
        }

        auto get_lca = [&](size_t u, size_t v) -> size_t 
        {
            while(depth[u] > depth[v]) 
            {
                u = parent[u];
            }

            while(depth[v] > depth[u]) 
            {
                v = parent[v];
            }

            while(u != v) 
            {
                u = parent[u];
                v = parent[v];
            }

            return u;
        };

        std::vector<std::vector<size_t>> edge_active_terms(numSites);
        std::vector<size_t> term_roots(OpTerms.size(), std::numeric_limits<size_t>::max());

        for(size_t t = 0UL; t < OpTerms.size(); ++t) 
        {
            const auto& term = OpTerms[t];

            size_t root_t = std::get<0>(term[0]);

            for(size_t k = 1UL; k < term.size(); ++k) 
            {
                root_t = get_lca(root_t, std::get<0>(term[k]));
            }
            
            term_roots[t] = root_t;

            for(const auto& op : term) 
            {
                size_t current = std::get<0>(op);

                while(current != root_t) 
                {
                    auto& active = edge_active_terms[current];
                    
                    if(std::find(active.begin(), active.end(), t) == active.end()) 
                    {
                        active.push_back(t);
                    }

                    current = parent[current];
                }
            }
        }

        virtualModesGraphType updated_graph;

        for(const auto& edge : graph) 
        {
            size_t u = std::get<0>(edge);
            size_t v = std::get<1>(edge);

            size_t child = (parent[u] == v) ? u : v;

            size_t D_q = 2UL + edge_active_terms[child].size();

            updated_graph.emplace(u, v, D_q);
        }

        graph = std::move(updated_graph);

        auto multiply_matrices = [](const std::vector<complexType>& A, 
                                    const std::vector<complexType>& B, 
                                    size_t d) -> std::vector<complexType>
        {
            std::vector<complexType> res(d * d, complexType(0.0, 0.0));

            for(size_t i = 0UL; i < d; ++i)
            {
                for(size_t j = 0UL; j < d; ++j)
                {
                    for(size_t k = 0UL; k < d; ++k)
                    {
                        res[i + j * d] += A[i + k * d] * B[k + j * d];
                    }
                }
            }   

            return res;
        };

        tensors_host.resize(numSites);

        for(size_t i = 0UL; i < numSites; ++i) 
        {
            size_t d = phys_dims[i];
            std::vector<complexType> Id(d * d, complexType(0.0, 0.0));

            for(size_t k = 0UL; k < d; ++k) 
            {
                Id[k + k * d] = complexType(1.0, 0.0);
            }

            size_t num_Q = neighbors_[i].size();
            size_t parent_edge_index = std::numeric_limits<size_t>::max();
            std::vector<size_t> D_q(num_Q);

            for(size_t q = 0UL; q < num_Q; ++q) 
            {
                size_t n = neighbors_[i][q];

                if(n == parent[i])
                {
                    parent_edge_index = q;
                }
                
                size_t child = (parent[i] == n) ? i : n;
                
                D_q[q] = 2UL + edge_active_terms[child].size();
            }

            size_t total_size = std::accumulate(D_q.begin(), 
                                                D_q.end(), 
                                                d * d, 
                                                [](size_t acc, const size_t& cur) {return acc * cur;});

            tensors_host[i].assign(total_size, complexType(0.0, 0.0));

            size_t num_children = (parent_edge_index == std::numeric_limits<size_t>::max()) ? num_Q : num_Q - 1;

            auto get_mapped_state = [&](size_t q, size_t semantic_state) -> size_t 
            {
                if(semantic_state <= 1UL)
                {
                    return semantic_state;
                }

                size_t global_t = semantic_state - 2UL;
                size_t n = neighbors_[i][q];
                size_t child = (parent[i] == n) ? i : n;

                const auto& active = edge_active_terms[child];
                auto it = std::find(active.begin(), active.end(), global_t);

                if(it != active.end()) 
                {
                    return 2UL + std::distance(active.begin(), it);
                }
                else 
                {
                    return std::numeric_limits<size_t>::max(); 
                }
            };

            auto add_tensor = [&](const std::vector<size_t>& child_states_semantic, 
                                  size_t out_state_semantic, 
                                  const std::vector<complexType>& M) 
            {
                if(parent_edge_index == std::numeric_limits<size_t>::max() && out_state_semantic != 1UL)
                {
                    return;
                }

                std::vector<size_t> Qmodes(num_Q);
                size_t child_idx = 0UL;

                for(size_t q = 0UL; q < num_Q; ++q) 
                {
                    size_t ms = get_mapped_state(q, (q == parent_edge_index) ? out_state_semantic : child_states_semantic[child_idx++]);

                    if(ms == std::numeric_limits<size_t>::max())
                    {
                        return;
                    }

                    Qmodes[q] = ms;
                }

                size_t offset = 0UL;
                size_t stride = d * d;

                for(size_t q = 0UL; q < num_Q; ++q) 
                {
                    offset += Qmodes[q] * stride;
                    stride *= D_q[q]; 
                }

                for(size_t p = 0UL; p < d * d; ++p) 
                {
                    tensors_host[i][p + offset] += M[p];
                }
            };

            std::vector<size_t> all_states(num_children, 0UL);

            if(num_children == 0UL) 
            {
                add_tensor({}, 0UL, Id);
            } 
            else 
            {
                add_tensor(all_states, 0UL, Id);

                for(size_t c = 0UL; c < num_children; ++c) 
                {
                    std::vector<size_t> v = all_states;

                    v[c] = 1UL;
                    add_tensor(v, 1UL, Id);
                }
            }

            for(size_t t = 0UL; t < OpTerms.size(); ++t) 
            {
                const auto& term = OpTerms[t];

                complexType coeff_total = {1.0, 0.0};
                std::vector<complexType> op_matrix = Id;
                bool is_in_term = false;

                for(const auto& op : term) 
                {
                    size_t site = std::get<0>(op);
                    complexType coeff = std::get<1>(op);
                    const auto& matrix = std::get<2>(op);

                    coeff_total *= coeff;

                    if(site == i) 
                    {
                        if(!is_in_term) 
                        {
                            op_matrix = matrix;
                            is_in_term = true;
                        } 
                        else
                        {
                            op_matrix = multiply_matrices(op_matrix, matrix, d);
                        }
                    }
                }

                std::vector<size_t> active_children;
                size_t child_idx = 0UL;

                for(size_t q = 0UL; q < num_Q; ++q) 
                {
                    if(q == parent_edge_index)
                    {
                        continue;
                    }

                    size_t child_node = neighbors_[i][q];
                    const auto& active = edge_active_terms[child_node];

                    if(std::find(active.begin(), active.end(), t) != active.end()) 
                    {
                        active_children.push_back(child_idx);
                    }
                    
                    child_idx++;
                }

                bool is_root = (term_roots[t] == i);

                if(is_in_term || !active_children.empty()) 
                {
                    std::vector<size_t> v = all_states;

                    for(size_t ac : active_children) 
                    {
                        v[ac] = 2UL + t;
                    }

                    size_t out_state = is_root ? 1UL : (2UL + t);

                    std::vector<complexType> M = op_matrix;

                    if(is_root) 
                    {
                        for(auto& val : M)
                        {
                            val *= coeff_total;
                        }
                    }

                    add_tensor(v, out_state, M);
                }
            }
        }
    }
    
    bool TensorNetwork::check_ = true;
    
    TensorNetwork::TensorNetwork(const std::vector<std::vector<int64_t>>& physExtents,
                                 const virtualModesGraphType& graph, size_t root, int64_t maxVirtualExtent, 
                                 size_t numStreams, size_t workSpaceLimit)
    {  
        if(maxVirtualExtent < 1L)
        {
            throw std::invalid_argument("TensorNetwork::TensorNetwork: "
                                         "maxVirtualExtent cannot be less than one.");
        }

        if(numStreams < 1UL)
        {
            throw std::invalid_argument("TensorNetwork::TensorNetwork: "
                                        "numStreams cannot be less than one.");
        }

        if(physExtents.size() < 2UL)
        {
            throw std::invalid_argument("TensorNetwork::TensorNetwork: "
                                        "The number of nodes cannot be less than two.");
        }

        if(graph.empty())
        {
            throw std::invalid_argument("TensorNetwork::TensorNetwork: "
                                        "The graph cannot be empty.");
        }

        if(workSpaceLimit < 1UL)
        {
            throw std::invalid_argument("TensorNetwork::TensorNetwork: "
                                        "The workSpaceLimit value must be at least 1 MB.");
        }

        maxVirtualExtent_ = maxVirtualExtent;
        numStreams_ = numStreams;
        numSites_ = physExtents.size();

        workSpaceLimit_ = workSpaceLimit * 1024UL * 1024UL;

        handle_ = std::vector<cutensornetHandle_t>(numStreams_, nullptr);
        svdConfig_ = std::vector<cutensornetTensorSVDConfig_t>(numStreams_, nullptr);
        streams_ = std::vector<cudaStream_t>(numStreams_, nullptr);

        globalMode_ = false;

        nodes_.resize(numSites_);

        for(size_t i = 0; i < numSites_; ++i)
        {
            if(physExtents[i].size() == 0)
            {
                throw std::runtime_error("TensorNetwork::TensorNetwork: "
                                         "Only single-hierarchy-level tensor networks are supported.");
            }
            
            for(size_t j = 0; j < physExtents[i].size(); ++j)
            {
                nodes_[i].physModes_.push_back(nextMode_++);
                nodes_[i].physExtents_.push_back(physExtents[i][j]);
            }
        }

        for(auto& edge : graph)
        {
            auto [in, out] = std::minmax(std::get<0UL>(edge), std::get<1UL>(edge));

            graph_.emplace(in, out);

            int64_t virtualExtent = std::get<2UL>(edge);

            if(virtualExtent < 1L) 
            {
                virtualExtent = 1L;
            }

            if((in < numSites_) && (out < numSites_))
            {
                int32_t virtualMode = nextMode_++;

                nodes_[in].virtualModes_.push_back(virtualMode);
                nodes_[out].virtualModes_.push_back(virtualMode);

                nodes_[in].virtualExtents_.push_back(virtualExtent);
                nodes_[out].virtualExtents_.push_back(virtualExtent);

                nodes_[in].neighbors_[out] = nodes_[in].virtualModes_.size() - 1UL;
                nodes_[out].neighbors_[in] = nodes_[out].virtualModes_.size() - 1UL;
            }
            else
            {
                throw std::invalid_argument("TensorNetwork::TensorNetwork: "
                                            "Site index can not exceed maximal number of sites.");
            }
        }

        graphTraversalToRoot_ = GetGraphTraversalToRoot(graph_, numSites_, root, loopFree_);

        for(size_t i = 0; i < numStreams_; ++i)
        {
            HANDLE_CUTN_ERROR(cutensornetCreate(&handle_[i]));
            HANDLE_CUTN_ERROR(cutensornetCreateTensorSVDConfig(handle_[i], &svdConfig_[i]));
            HANDLE_CUDA_ERROR(cudaStreamCreate(&streams_[i]));
        }

        SetSVDConfig(absCutoff_, 
                     relCutoff_, 
                     partition_, 
                     renorm_, 
                     svdAlgo_, 
                     svdParams_, 
                     svdParamsSize_);
    }

    TensorNetwork::TensorNetwork(const std::string& path)
    {
        Load(path);
    }

    TensorNetwork::~TensorNetwork()
    {
        ClearNet();
    }

    TensorNetwork::TensorNetwork(const TensorNetwork& other) : loopFree_(other.loopFree_),
                                                               globalMode_(other.globalMode_),
                                                               nodes_(other.nodes_),
                                                               graph_(other.graph_),
                                                               graphTraversalToRoot_(other.graphTraversalToRoot_),
                                                               numSites_(other.numSites_),
                                                               maxVirtualExtent_(other.maxVirtualExtent_),
                                                               nextMode_(other.nextMode_),
                                                               numStreams_(other.numStreams_),
                                                               workSpacePreference_(other.workSpacePreference_),
                                                               workSpaceLimit_(other.workSpaceLimit_)
    {
        tensors_ = std::vector<void*>(numSites_, nullptr);
        handle_ = std::vector<cutensornetHandle_t>(numStreams_, nullptr);
        streams_ = std::vector<cudaStream_t>(numStreams_, nullptr);
        svdConfig_ = std::vector<cutensornetTensorSVDConfig_t>(numStreams_, nullptr);

        for(size_t i = 0; i < numSites_; ++i)
        {
            size_t dim = other.GetTensorSize(i);

            void* tensor_device;

            HANDLE_CUDA_ERROR(cudaMalloc(&tensor_device, dim * sizeof(complexType)));
            HANDLE_CUDA_ERROR(cudaMemcpy(tensor_device, other.tensors_[i], dim * sizeof(complexType), cudaMemcpyDeviceToDevice));

            tensors_[i] = tensor_device;
        }

        for (size_t i = 0; i < numStreams_; ++i)
        {
            HANDLE_CUTN_ERROR(cutensornetCreate(&handle_[i]));
            HANDLE_CUTN_ERROR(cutensornetCreateTensorSVDConfig(handle_[i], &svdConfig_[i]));
            HANDLE_CUDA_ERROR(cudaStreamCreate(&streams_[i]));
        }

        SetSVDConfig(other.absCutoff_, 
                     other.relCutoff_,
                     other.partition_,  
                     other.renorm_, 
                     other.svdAlgo_, 
                     other.svdParams_, 
                     other.svdParamsSize_);
    }

    TensorNetwork::TensorNetwork(TensorNetwork&& other) noexcept : loopFree_(other.loopFree_),
                                                                   globalMode_(other.globalMode_),
                                                                   nodes_(std::move(other.nodes_)),
                                                                   graph_(std::move(other.graph_)),
                                                                   graphTraversalToRoot_(std::move(other.graphTraversalToRoot_)),
                                                                   numSites_(other.numSites_),
                                                                   maxVirtualExtent_(other.maxVirtualExtent_),
                                                                   tensors_(std::move(other.tensors_)),
                                                                   handle_(std::move(other.handle_)),
                                                                   streams_(std::move(other.streams_)),
                                                                   svdConfig_(std::move(other.svdConfig_)),
                                                                   nextMode_(other.nextMode_),
                                                                   numStreams_(other.numStreams_),
                                                                   workSpacePreference_(other.workSpacePreference_),
                                                                   workSpaceLimit_(other.workSpaceLimit_),
                                                                   absCutoff_(other.absCutoff_), 
                                                                   relCutoff_(other.relCutoff_), 
                                                                   partition_(other.partition_),
                                                                   renorm_(other.renorm_), 
                                                                   svdAlgo_(other.svdAlgo_), 
                                                                   svdParams_(other.svdParams_), 
                                                                   svdParamsSize_(other.svdParamsSize_)
    {
        other.numSites_ = 0;
        other.maxVirtualExtent_ = 1;
        other.nextMode_ = 0;
        other.numStreams_ = 1;
        other.svdParams_ = nullptr;
        other.svdParamsSize_ = 0;
        globalMode_ = false;
    }

    TensorNetwork& TensorNetwork::operator=(const TensorNetwork& other)
    {
        if (this != &other)
        {
            ClearNet();
            
            loopFree_ = other.loopFree_;
            globalMode_ = other.globalMode_;
            nodes_ = other.nodes_;
            graph_ = other.graph_;
            graphTraversalToRoot_ = other.graphTraversalToRoot_;
            numSites_ = other.numSites_;
            maxVirtualExtent_ = other.maxVirtualExtent_;
            nextMode_ = other.nextMode_;
            numStreams_ = other.numStreams_;
            workSpacePreference_ = other.workSpacePreference_;
            workSpaceLimit_ = other.workSpaceLimit_;
            
            tensors_ = std::vector<void*>(numSites_, nullptr);
            handle_ = std::vector<cutensornetHandle_t>(numStreams_, nullptr);
            streams_ = std::vector<cudaStream_t>(numStreams_, nullptr);
            svdConfig_ = std::vector<cutensornetTensorSVDConfig_t>(numStreams_, nullptr);

            for(size_t i = 0; i < numSites_; ++i)
            {
                size_t dim = other.GetTensorSize(i);
    
                void* tensor_device;

                HANDLE_CUDA_ERROR(cudaMalloc(&tensor_device, dim * sizeof(complexType)));
                HANDLE_CUDA_ERROR(cudaMemcpy(tensor_device, other.tensors_[i], dim * sizeof(complexType), cudaMemcpyDeviceToDevice));

                tensors_[i] = tensor_device;
            }

            for (size_t i = 0; i < numStreams_; ++i)
            {
                HANDLE_CUTN_ERROR(cutensornetCreate(&handle_[i]));
                HANDLE_CUTN_ERROR(cutensornetCreateTensorSVDConfig(handle_[i], &svdConfig_[i]));
                HANDLE_CUDA_ERROR(cudaStreamCreate(&streams_[i]));
            }

            SetSVDConfig(other.absCutoff_, 
                         other.relCutoff_,
                         other.partition_, 
                         other.renorm_,  
                         other.svdAlgo_, 
                         other.svdParams_, 
                         other.svdParamsSize_);
        }

        return *this;
    }

    TensorNetwork& TensorNetwork::operator=(TensorNetwork&& other) noexcept
    {
        if (this != &other)
        {
            ClearNet();
            
            loopFree_ = other.loopFree_;
            globalMode_ = other.globalMode_;
            nodes_ = std::move(other.nodes_);
            graph_ = std::move(other.graph_);
            graphTraversalToRoot_ = std::move(other.graphTraversalToRoot_);
            numSites_ = other.numSites_;
            maxVirtualExtent_ = other.maxVirtualExtent_;
            tensors_ = std::move(other.tensors_);
            handle_ = std::move(other.handle_);
            streams_ = std::move(other.streams_);
            svdConfig_ = std::move(other.svdConfig_);
            nextMode_ = other.nextMode_;
            numStreams_ = other.numStreams_;
            workSpacePreference_ = other.workSpacePreference_;
            workSpaceLimit_ = other.workSpaceLimit_;
            absCutoff_ = other.absCutoff_;
            relCutoff_ = other.relCutoff_; 
            partition_ = other.partition_;
            renorm_ = other.renorm_;
            svdAlgo_ = other.svdAlgo_;
            svdParams_ = other.svdParams_; 
            svdParamsSize_ = other.svdParamsSize_;

            other.numSites_ = 0;
            other.maxVirtualExtent_ = 1;
            other.nextMode_ = 0;
            other.numStreams_ = 1;
            other.svdParams_ = nullptr;
            other.svdParamsSize_ = 0;
            globalMode_ = false;
        }

        return *this;
    }

    void TensorNetwork::Save(const std::string& path)
    {
        std::filesystem::path dirPath(path);
    
        if(!std::filesystem::exists(dirPath)) 
        {
            if(!std::filesystem::create_directories(dirPath)) 
            {
                throw std::runtime_error("TensorNetwork::Save: "
                                         "The directory " + path + " can not be created.");
            }
        } 
        else if(!std::filesystem::is_directory(dirPath)) 
        {
            throw std::invalid_argument("TensorNetwork::Save: "
                                        "The path " + path + " exists but is not a directory.");
        }
    
        std::filesystem::path metaPath = dirPath / "MetaData.yaml";
        std::filesystem::path tensorsPath = dirPath / "Tensors.bin";

        YAML::Node yaml_node;

        yaml_node["loopFree"] = loopFree_;
        yaml_node["globalMode"] = globalMode_;
        yaml_node["numSites"] = numSites_;
        yaml_node["maxVirtualExtent"] = maxVirtualExtent_;
        yaml_node["nextMode"] = nextMode_;
        yaml_node["numStreams"] = numStreams_;
        yaml_node["workSpaceLimit"] = workSpaceLimit_;
        yaml_node["absCutoff"] = absCutoff_;
        yaml_node["relCutoff"] = relCutoff_;

        switch(workSpacePreference_)
        {
            case CUTENSORNET_WORKSIZE_PREF_MIN: 
                yaml_node["workSpacePreference"] = "CUTENSORNET_WORKSIZE_PREF_MIN";
                break;
            case CUTENSORNET_WORKSIZE_PREF_RECOMMENDED:
                yaml_node["workSpacePreference"] = "CUTENSORNET_WORKSIZE_PREF_RECOMMENDED";
                break;
            case CUTENSORNET_WORKSIZE_PREF_MAX:
                yaml_node["workSpacePreference"] = "CUTENSORNET_WORKSIZE_PREF_MAX";
                break;
        }

        switch(partition_)
        {
            case CUTENSORNET_TENSOR_SVD_PARTITION_NONE: 
                yaml_node["partition"] = "CUTENSORNET_TENSOR_SVD_PARTITION_NONE";
                break;
            case CUTENSORNET_TENSOR_SVD_PARTITION_US:
                yaml_node["partition"] = "CUTENSORNET_TENSOR_SVD_PARTITION_US";
                break;
            case CUTENSORNET_TENSOR_SVD_PARTITION_SV:
                yaml_node["partition"] = "CUTENSORNET_TENSOR_SVD_PARTITION_SV";
                break;
            case CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL:
                yaml_node["partition"] = "CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL";
                break;
        }

        switch(renorm_)
        {
            case CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE: 
                yaml_node["renorm"] = "CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE";
                break;
            case CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1:
                yaml_node["renorm"] = "CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1";
                break;
            case CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2:
                yaml_node["renorm"] = "CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2";
                break;
            case CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF:
                yaml_node["renorm"] = "CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF";
                break;
        }

        switch(svdAlgo_)
        {
            case CUTENSORNET_TENSOR_SVD_ALGO_GESVD:
                yaml_node["svdAlgo"] = "CUTENSORNET_TENSOR_SVD_ALGO_GESVD";
                break;
            case CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ:
                yaml_node["svdAlgo"] = "CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ"; 
                if((svdParamsSize_ > 0UL) && (svdParams_ != nullptr))
                {
                    auto params = static_cast<cutensornetGesvdjParams_t*>(svdParams_);
                    yaml_node["svdAlgo_tol"] = params->tol;
                    yaml_node["svdAlgo_maxSweeps"] = params->maxSweeps;
                }
                break;
            case CUTENSORNET_TENSOR_SVD_ALGO_GESVDP:
                yaml_node["svdAlgo"] = "CUTENSORNET_TENSOR_SVD_ALGO_GESVDP";
                break;
            case CUTENSORNET_TENSOR_SVD_ALGO_GESVDR:
                yaml_node["svdAlgo"] = "CUTENSORNET_TENSOR_SVD_ALGO_GESVDR";
                if((svdParamsSize_ > 0UL) && (svdParams_ != nullptr))
                {
                    auto params = static_cast<cutensornetGesvdrParams_t*>(svdParams_);
                    yaml_node["svdAlgo_oversampling"] = params->oversampling;
                    yaml_node["svdAlgo_niters"] = params->niters;
                }
                break;
        }

        for(const auto& edge : graph_) 
        {
            yaml_node["graph"].push_back(edge);
        }

        yaml_node["graph"].SetStyle(YAML::EmitterStyle::value::Flow);

        for (const auto& edge : graphTraversalToRoot_) 
        {
            yaml_node["graphTraversalToRoot"].push_back(edge);
        }

        yaml_node["graphTraversalToRoot"].SetStyle(YAML::EmitterStyle::value::Flow);

        for(size_t i = 0; i < nodes_.size(); ++i) 
        {
            const auto& node = nodes_[i];
            auto local_node = yaml_node["nodes"][i];

            local_node["physModes"] = node.physModes_;
            local_node["physModes"].SetStyle(YAML::EmitterStyle::value::Flow);
            
            local_node["physExtents"] = node.physExtents_;
            local_node["physExtents"].SetStyle(YAML::EmitterStyle::value::Flow);

            local_node["virtualModes"] = node.virtualModes_;
            local_node["virtualModes"].SetStyle(YAML::EmitterStyle::value::Flow);

            local_node["virtualExtents"] = node.virtualExtents_;
            local_node["virtualExtents"].SetStyle(YAML::EmitterStyle::value::Flow);

            local_node["extra_virtualModes"] = node.extra_virtualModes_;
            local_node["extra_virtualModes"].SetStyle(YAML::EmitterStyle::value::Flow);

            local_node["extra_virtualExtents"] = node.extra_virtualExtents_;
            local_node["extra_virtualExtents"].SetStyle(YAML::EmitterStyle::value::Flow);

            local_node["neighbors"] = node.neighbors_;
            local_node["neighbors"].SetStyle(YAML::EmitterStyle::value::Flow);
        }

        std::ofstream ofsMeta(metaPath, std::ios::trunc);

        if(!ofsMeta) 
        {
            throw std::runtime_error("TensorNetwork::Save: "
                                     "The file " + metaPath.string() + " can not be opened for writing.");
        }

        ofsMeta << yaml_node;

        ofsMeta.close();
        
        std::ofstream ofsTensors(tensorsPath, std::ios::binary | std::ios::trunc);

        for (size_t id = 0; id < numSites_; ++id) 
        {
            std::vector<complexType> data = GetTensorData(id);
  
            ofsTensors.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(complexType));
        }

        ofsTensors.close();
    }
    
    void TensorNetwork::Load(const std::string& path)
    {
        std::filesystem::path dir(path);

        if(!std::filesystem::exists(dir)) 
        {
            throw std::invalid_argument("TensorNetwork::Load: "
                                        "The directory " + path + " does not exist.");
        }

        if(!std::filesystem::is_directory(dir)) 
        {
            throw std::invalid_argument("TensorNetwork::Load: "
                                        "The path " + path + " is not a directory.");
        }

        std::filesystem::path metaPath = dir / "MetaData.yaml";
        std::filesystem::path tensorsPath = dir / "Tensors.bin";

        if(!std::filesystem::exists(metaPath) || !std::filesystem::exists(tensorsPath)) 
        {
            throw std::invalid_argument("TensorNetwork::Load: "
                                        "Required files (MetaData.yaml, Tensors.bin) were not found in " + path + ".");
        }

        ClearNet();

        YAML::Node yaml_node;
        
        try 
        {
            yaml_node = YAML::LoadFile(metaPath.string());
        }
        catch (const YAML::Exception& ex) 
        {
            throw std::runtime_error("TensorNetwork::Load: "
                                     "Failed to parse YAML file " + metaPath.string() + ": " + ex.what());
        }

        loopFree_ = yaml_node["loopFree"].as<bool>();
        globalMode_ = yaml_node["globalMode"].as<bool>();
        numSites_ = yaml_node["numSites"].as<size_t>();
        maxVirtualExtent_ = yaml_node["maxVirtualExtent"].as<int64_t>();
        nextMode_ = yaml_node["nextMode"].as<int32_t>();
        numStreams_ = yaml_node["numStreams"].as<size_t>();
        workSpaceLimit_ = yaml_node["workSpaceLimit"].as<size_t>();

        double absCutoff = yaml_node["absCutoff"].as<double>();
        double relCutoff = yaml_node["relCutoff"].as<double>();

        std::string prefStr = yaml_node["workSpacePreference"].as<std::string>();

        if(prefStr == "CUTENSORNET_WORKSIZE_PREF_MIN")
        {
            workSpacePreference_ = CUTENSORNET_WORKSIZE_PREF_MIN;
        }
        else if(prefStr == "CUTENSORNET_WORKSIZE_PREF_RECOMMENDED")
        {
            workSpacePreference_ = CUTENSORNET_WORKSIZE_PREF_RECOMMENDED;
        }
        else if(prefStr == "CUTENSORNET_WORKSIZE_PREF_MAX")
        {
            workSpacePreference_ = CUTENSORNET_WORKSIZE_PREF_MAX;
        }
        else
        {
            throw std::runtime_error("TensorNetwork::Load: "
                                     "Unknown workSpacePreference value: " + prefStr);
        }

        cutensornetTensorSVDPartition_t partition;
        std::string partStr = yaml_node["partition"].as<std::string>();

        if(partStr == "CUTENSORNET_TENSOR_SVD_PARTITION_NONE")
        {
            partition = CUTENSORNET_TENSOR_SVD_PARTITION_NONE;
        }
        else if(partStr == "CUTENSORNET_TENSOR_SVD_PARTITION_US")
        {
            partition = CUTENSORNET_TENSOR_SVD_PARTITION_US;
        }
        else if(partStr == "CUTENSORNET_TENSOR_SVD_PARTITION_SV")
        {
            partition = CUTENSORNET_TENSOR_SVD_PARTITION_SV;
        }
        else if(partStr == "CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL")
        {
            partition = CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL;
        }
        else
        {
            throw std::runtime_error("TensorNetwork::Load: "
                                     "Unknown partition value: " + partStr);
        }

        cutensornetTensorSVDNormalization_t renorm;
        std::string renormStr = yaml_node["renorm"].as<std::string>();

        if(renormStr == "CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE")
        {
            renorm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE;
        }
        else if(renormStr == "CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1")
        {
            renorm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L1;
        }
        else if(renormStr == "CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2")
        {
            renorm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2;
        }
        else if(renormStr == "CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF")
        {
            renorm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_LINF;
        }
        else
        {
            throw std::runtime_error("TensorNetwork::Load: "
                                     "Unknown renorm value: " + renormStr);
        }

        cutensornetTensorSVDAlgo_t svdAlgo;
        std::string algoStr = yaml_node["svdAlgo"].as<std::string>();

        void* svdParams = nullptr;
        size_t svdParamsSize = 0;

        if(algoStr == "CUTENSORNET_TENSOR_SVD_ALGO_GESVD") 
        {
            svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
        } 
        else if(algoStr == "CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ") 
        {
            svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;

            if(yaml_node["svdAlgo_tol"] && yaml_node["svdAlgo_maxSweeps"]) 
            {
                auto params = new cutensornetGesvdjParams_t;

                params->tol = yaml_node["svdAlgo_tol"].as<double>();
                params->maxSweeps = yaml_node["svdAlgo_maxSweeps"].as<int32_t>();

                svdParams = static_cast<void*>(params);
                svdParamsSize = sizeof(cutensornetGesvdjParams_t);
            }
        } 
        else if(algoStr == "CUTENSORNET_TENSOR_SVD_ALGO_GESVDP") 
        {
            svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDP;
        } 
        else if(algoStr == "CUTENSORNET_TENSOR_SVD_ALGO_GESVDR") 
        {
            svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDR;

            if(yaml_node["svdAlgo_oversampling"] && yaml_node["svdAlgo_niters"]) 
            {
                auto params = new cutensornetGesvdrParams_t;

                params->oversampling = yaml_node["svdAlgo_oversampling"].as<int64_t>();
                params->niters = yaml_node["svdAlgo_niters"].as<int64_t>();

                svdParams = static_cast<void*>(params);
                svdParamsSize = sizeof(cutensornetGesvdrParams_t);
            }
        } 
        else 
        {
            throw std::runtime_error("TensorNetwork::Load: "
                                     "Unknown svdAlgo value: " + algoStr);
        }

        for(const auto& edgeNode : yaml_node["graph"]) 
        {
            size_t u = edgeNode[0].as<size_t>();
            size_t v = edgeNode[1].as<size_t>();

            graph_.insert({u, v});
        }

        for(const auto& edgeNode : yaml_node["graphTraversalToRoot"]) 
        {
            size_t u = edgeNode[0].as<size_t>();
            size_t v = edgeNode[1].as<size_t>();

            graphTraversalToRoot_.emplace_back(u, v);
        }

        for(size_t i = 0; i < yaml_node["nodes"].size(); ++i) 
        {
            YAML::Node nodeNode = yaml_node["nodes"][i];
            Node node;

            node.physModes_ = nodeNode["physModes"].as<std::vector<int32_t>>();
            node.physExtents_ = nodeNode["physExtents"].as<std::vector<int64_t>>();
            node.virtualModes_ = nodeNode["virtualModes"].as<std::vector<int32_t>>();
            node.virtualExtents_ = nodeNode["virtualExtents"].as<std::vector<int64_t>>();
            node.extra_virtualModes_ = nodeNode["extra_virtualModes"].as<std::vector<int32_t>>();
            node.extra_virtualExtents_ = nodeNode["extra_virtualExtents"].as<std::vector<int64_t>>();
            node.neighbors_ = nodeNode["neighbors"].as<std::unordered_map<size_t, size_t>>();

            nodes_.push_back(std::move(node));
        }

        std::ifstream ifsTensors(tensorsPath, std::ios::binary);

        if(!ifsTensors) 
        {
            throw std::runtime_error("TensorNetwork::Load: "
                                     "The file " + tensorsPath.string() + "can not be opened.");
        }

        for (size_t id = 0; id < numSites_; ++id) 
        {
            size_t ten_size = GetTensorSize(id);

            std::vector<complexType> data(ten_size);

            ifsTensors.read(reinterpret_cast<char*>(data.data()), ten_size * sizeof(complexType));

            if (ifsTensors.gcount() != static_cast<std::streamsize>(ten_size * sizeof(complexType))) 
            {
                throw std::runtime_error("TensorNetwork::Load: "
                                         "Unexpected end of file while reading tensor " + std::to_string(id) + 
                                         " (expected " + std::to_string(ten_size * sizeof(complexType)) + " bytes, "
                                         "got " + std::to_string(ifsTensors.gcount()) + ").");
            }

            SetTensorData(id, data);
        }

        ifsTensors.close();

        handle_ = std::vector<cutensornetHandle_t>(numStreams_, nullptr);
        streams_ = std::vector<cudaStream_t>(numStreams_, nullptr);
        svdConfig_ = std::vector<cutensornetTensorSVDConfig_t>(numStreams_, nullptr);

        for (size_t i = 0; i < numStreams_; ++i)
        {
            HANDLE_CUTN_ERROR(cutensornetCreate(&handle_[i]));
            HANDLE_CUTN_ERROR(cutensornetCreateTensorSVDConfig(handle_[i], &svdConfig_[i]));
            HANDLE_CUDA_ERROR(cudaStreamCreate(&streams_[i]));
        }

        SetSVDConfig(absCutoff, 
                     relCutoff,
                     partition, 
                     renorm,  
                     svdAlgo, 
                     svdParams, 
                     svdParamsSize);

        free(svdParams);
    }

    void TensorNetwork::SetSVDConfig(double absCutoff, 
                                     double relCutoff, 
                                     cutensornetTensorSVDPartition_t partition,
                                     cutensornetTensorSVDNormalization_t renorm,
                                     cutensornetTensorSVDAlgo_t svdAlgo,
                                     const void* svdParams, size_t svdParamsSize)
    {        
        if((absCutoff < 0.0) || (absCutoff >= 1.0))
        {
            throw std::invalid_argument("TensorNetwork::SetSVDConfig: "
                                        "absCutoff must be in the range [0, 1)");
        }

        if((relCutoff < 0.0) || (relCutoff >= 1.0))
        {
            throw std::invalid_argument("TensorNetwork::SetSVDConfig: "
                                        "relCutoff must be in the range [0, 1)");
        }
        
        absCutoff_ = absCutoff;
        relCutoff_ = relCutoff;

        partition_ = partition;
        renorm_ = renorm;
        svdAlgo_ = svdAlgo;

        if((svdParams != nullptr) && (svdParamsSize > 0))
        {
            svdParamsSize_ = svdParamsSize;
            
            if(svdParams_ != nullptr)
            {
                free(svdParams_);
            }
            
            svdParams_ = malloc(svdParamsSize_);

            memcpy(svdParams_, svdParams, svdParamsSize_);
        }
        
        for(size_t i = 0; i < numStreams_; ++i)
        {
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_[i], 
                                                                     svdConfig_[i], 
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF, 
                                                                     &absCutoff_, 
                                                                     sizeof(absCutoff_)));
            
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_[i], 
                                                                     svdConfig_[i], 
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, 
                                                                     &relCutoff_, 
                                                                     sizeof(relCutoff_)));

            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_[i], 
                                                                     svdConfig_[i],
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, 
                                                                     &partition_, 
                                                                     sizeof(partition_)));
            
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_[i], 
                                                                     svdConfig_[i], 
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_S_NORMALIZATION, 
                                                                     &renorm_, 
                                                                     sizeof(renorm_)));

            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_[i], 
                                                                     svdConfig_[i],
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, 
                                                                     &svdAlgo_, 
                                                                     sizeof(svdAlgo_)));
            
            if((svdParams_ != nullptr) && (svdParamsSize_ > 0))
            {
                HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_[i], 
                                                                         svdConfig_[i],
                                                                         CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, 
                                                                         svdParams_, 
                                                                         svdParamsSize_));
            }
        }
    }

    void TensorNetwork::SetMaxVirtualExtent(int64_t val)
    {
        if(val < 1L)
        {
            throw std::invalid_argument("TensorNetwork::SetMaxVirtualExtent: "
                                         "maxVirtualExtent cannot be less than one.");
        }
        
        maxVirtualExtent_ = val;
    }

    void TensorNetwork::SetWorkSpacePreference(cutensornetWorksizePref_t pref)
    {
        workSpacePreference_ = pref;
    }

    void TensorNetwork::SetWorkSpaceLimit(size_t val)
    {
        if(val < 1UL)
        {
            throw std::invalid_argument("TensorNetwork::SetWorkSpaceLimit: "
                                        "The workSpaceLimit value must be at least 1 MB.");
        }
        
        workSpaceLimit_ = val * 1024UL * 1024UL;
    }

    void TensorNetwork::SetGlobalMode(bool mode)
    {
        globalMode_ = mode;
    }

    const Node& TensorNetwork::GetNode(size_t id) const
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetNode: "
                                        "Site index can not exceed maximal number of sites.");
        }

        return nodes_[id];
    }

    Node& TensorNetwork::GetNode(size_t id)
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetNode: "
                                        "Site index can not exceed maximal number of sites.");
        }

        return nodes_[id];
    }

    size_t TensorNetwork::GetNumSites() const
    {
        return nodes_.size();
    }

    size_t TensorNetwork::GetTensorSize(size_t id, bool include_extra) const
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetTensorSize: "
                                        "Site index can not exceed maximal number of sites.");
        }

        const auto& node = nodes_[id];
        
        int64_t dim = std::accumulate(node.physExtents_.begin(),
                                      node.physExtents_.end(),
                                      1L,
                                      [](int64_t acc, const int64_t& current) {return acc * current;});

        dim = std::accumulate(node.virtualExtents_.begin(),
                              node.virtualExtents_.end(),
                              dim,
                              [](int64_t acc, const int64_t& current) {return acc * current;});
        
        if(include_extra)
        {
            if(!node.extra_virtualExtents_.empty())
            {
                dim = std::accumulate(node.extra_virtualExtents_.begin(),
                                      node.extra_virtualExtents_.end(),
                                      dim,
                                      [](int64_t acc, const int64_t& current) {return acc * current;});
            }
        }

        return static_cast<size_t>(dim);
    }

    std::vector<int32_t> TensorNetwork::GetTensorModes(size_t id, bool include_extra) const
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetTensorModes: "
                                        "Site index can not exceed maximal number of sites.");
        }
        
        const auto& node = nodes_[id];
        
        std::vector<int32_t> modes = node.physModes_;
        modes.insert(modes.end(), node.virtualModes_.begin(), node.virtualModes_.end());
        
        if(include_extra)
        {
            modes.insert(modes.end(), node.extra_virtualModes_.begin(), node.extra_virtualModes_.end());
        }

        return modes;
    }

    std::vector<int64_t> TensorNetwork::GetTensorExtents(size_t id, bool include_extra) const
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetTensorExtents: "
                                        "Site index can not exceed maximal number of sites.");
        }
        
        const auto& node = nodes_[id];

        std::vector<int64_t> extents = node.physExtents_;
        extents.insert(extents.end(), node.virtualExtents_.begin(), node.virtualExtents_.end());
        
        if(include_extra)
        {
            extents.insert(extents.end(), node.extra_virtualExtents_.begin(), node.extra_virtualExtents_.end());
        }

        return extents;
    }

    void TensorNetwork::SetState(const std::vector<std::vector<complexType>>& tensors_host)
    {
        if(!tensors_.empty())
        {
            ClearTensors();
        }

        if(tensors_host.size() != numSites_)
        {
            throw std::invalid_argument("TensorNetwork::SetInitialState: "
                                        "The size of the vector of host tensors should be equal to numSites_.");
        }

        for(size_t i = 0; i < numSites_; ++i)
        {
            size_t dim = GetTensorSize(i);

            size_t ht_size = tensors_host[i].size();

            if(dim != ht_size)
            {
                throw std::invalid_argument("TensorNetwork::SetInitialState: "
                                            "Incorrect number of tensors_host[" + std::to_string(i) + "] elements.");
            }
            
            void* tensor_device;

            HANDLE_CUDA_ERROR(cudaMalloc(&tensor_device, ht_size * sizeof(complexType)));
            HANDLE_CUDA_ERROR(cudaMemcpy(tensor_device, (void*)(tensors_host[i].data()), ht_size * sizeof(complexType), cudaMemcpyHostToDevice));

            tensors_.push_back(tensor_device);
        }
    }

    void TensorNetwork::ClearTensors()
    {
        for(auto ten : tensors_)
        {
            if(ten != nullptr) 
            {
                HANDLE_CUDA_ERROR(cudaFree(ten));
            }
        }

        tensors_.clear();
    }

    void TensorNetwork::ClearNet()
    {
        nextMode_ = 0;
        
        nodes_.clear();
        graph_.clear();
        graphTraversalToRoot_.clear();

        ClearTensors();
        
        for(auto it : handle_)
        {
            HANDLE_CUTN_ERROR(cutensornetDestroy(it));
        }

        handle_.clear();

        for(auto it : streams_)
        {
            HANDLE_CUDA_ERROR(cudaStreamDestroy(it));
        }

        streams_.clear();

        for(auto it : svdConfig_)
        {
            HANDLE_CUTN_ERROR(cutensornetDestroyTensorSVDConfig(it));
        }

        svdConfig_.clear();

        if(svdParams_ != nullptr)
        {
            free(svdParams_);
        }

        svdParams_ = nullptr;
        svdParamsSize_ = 0UL;
    }

    int32_t TensorNetwork::GetNextMode(bool update)
    {
        if(update)
        {
            return nextMode_++;
        }
        else
        {
            return nextMode_;
        }
    }

    cutensornetHandle_t TensorNetwork::GetHandle(size_t id)
    {
        if(id >= numStreams_)
        {
            throw std::invalid_argument("TensorNetwork::GetHandle: "
                                        "Stream index can not exceed maximal number of streams.");
        }
        
        return handle_[id];
    }
    
    cudaStream_t TensorNetwork::GetStream(size_t id)
    {
        if(id >= numStreams_)
        {
            throw std::invalid_argument("TensorNetwork::GetStream: "
                                        "Stream index can not exceed maximal number of streams.");
        }
        
        return streams_[id];
    }

    std::vector<complexType> TensorNetwork::GetTensorData(size_t id) const
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetTensorData: "
                                        "Site index can not exceed maximal number of sites.");
        }

        if(tensors_.empty()) 
        {
            throw std::runtime_error("TensorNetwork::GetTensorData: "
                                     "The state is not initialized.");
        }

        size_t dim = GetTensorSize(id);

        std::vector<complexType> tensor_host(dim);

        HANDLE_CUDA_ERROR(cudaMemcpy((void*)(tensor_host.data()), tensors_[id], dim * sizeof(complexType), cudaMemcpyDeviceToHost));

        return tensor_host;
    }

    void TensorNetwork::SetTensorData(size_t id, const std::vector<complexType>& ten)
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::SetTensorData: "
                                        "Site index can not exceed maximal number of sites.");
        }

        if(tensors_.empty()) 
        {
            tensors_ = std::vector<void*>(numSites_, nullptr);
        }

        size_t ht_size = ten.size();

        if(GetTensorSize(id) != ht_size)
        {
            throw std::invalid_argument("TensorNetwork::SetTensorData: "
                                        "Incorrect number of elements of the ten vector.");
        }
        
        void* tensor_device;

        HANDLE_CUDA_ERROR(cudaMalloc(&tensor_device, ht_size * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemcpy(tensor_device, (void*)(ten.data()), ht_size * sizeof(complexType), cudaMemcpyHostToDevice));

        if(tensors_[id] != nullptr)
        {
            HANDLE_CUDA_ERROR(cudaFree(tensors_[id]));
        }

        tensors_[id] = tensor_device;
    }

    const void* TensorNetwork::GetTensor(size_t id) const
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::GetTensor: "
                                        "Site index can not exceed maximal number of sites.");
        }

        if(tensors_.empty()) 
        {
            throw std::runtime_error("TensorNetwork::GetTensor: "
                                     "The state is not initialized.");
        }

        return tensors_[id];
    }

    void TensorNetwork::SetTensor(size_t id, void* ten, size_t ten_size)
    {
        if(id >= numSites_)
        {
            throw std::invalid_argument("TensorNetwork::SetTensor: "
                                        "Site index can not exceed maximal number of sites.");
        }

        if(tensors_.empty()) 
        {
            tensors_ = std::vector<void*>(numSites_, nullptr);
        }

        if(GetTensorSize(id) != ten_size)
        {
            throw std::invalid_argument("TensorNetwork::SetTensor: "
                                        "Incorrect number of elements of the ten vector.");
        }

        if(tensors_[id] != nullptr)
        {
            HANDLE_CUDA_ERROR(cudaFree(tensors_[id]));
        }

        tensors_[id] = ten;
    }

    void TensorNetwork::SynchronizeStreams(const std::vector<size_t>& ids)
    {
        if(ids.empty())
        {
            for(size_t i = 0; i < numStreams_; ++i)
            {
                HANDLE_CUDA_ERROR(cudaStreamSynchronize(streams_[i]));
            }
        }
        else
        {
            for(const auto& it : ids)
            {
                if(it < numStreams_)
                {
                    HANDLE_CUDA_ERROR(cudaStreamSynchronize(streams_[it]));
                }
            }
        }
    }

    void* TensorNetwork::ComputeTwoSiteVector(size_t siteA, 
                                              size_t siteB,
                                              bool conjugate,
                                              size_t thread_num)
    {
        if(check_)
        {
            if((siteB >= numSites_) || (siteA >= numSites_))
            {
                throw std::invalid_argument("TensorNetwork::ComputeTwoSiteVector: "
                                            "Site index can not exceed maximal number of sites.");
            }

            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ComputeTwoSiteVector: "
                                         "The state is not initialized.");
            }
        }

        auto& siteA_node = nodes_[siteA];
        auto& siteB_node = nodes_[siteB];

        auto itA = siteA_node.neighbors_.find(siteB);
        auto itB = siteB_node.neighbors_.find(siteA);
        
        if(check_)
        {
            if((itA == siteA_node.neighbors_.end()) || (itB == siteB_node.neighbors_.end()))
            {
                throw std::invalid_argument("TensorNetwork::ComputeTwoSiteVector: "
                                             "Site " + std::to_string(siteA) + " must be a neighbor of site " 
                                             + std::to_string(siteB) + ".");
            }

            if(!siteA_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::ComputeTwoSiteVector: "
                                         "Duplicate bonds should be excluded for site " 
                                         + std::to_string(siteA) + ".");
            }

            if(!siteB_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::ComputeTwoSiteVector: "
                                         "Duplicate bonds should be excluded for site " 
                                         + std::to_string(siteB) + ".");
            }
        }

        size_t indexA = itA->second;
        size_t indexB = itB->second;

        std::vector<std::vector<int32_t>> modesInAB(2);
        std::vector<std::vector<int64_t>> extentsInAB(2);
            
        modesInAB[0] = GetTensorModes(siteA, false);
        extentsInAB[0] = GetTensorExtents(siteA, false);

        modesInAB[1] = GetTensorModes(siteB, false);
        extentsInAB[1] = GetTensorExtents(siteB, false);

        std::vector<const void*> tensorsInAB(2);
            
        tensorsInAB[0] = tensors_[siteA];
        tensorsInAB[1] = tensors_[siteB];

        std::vector<cutensornetTensorQualifiers_t> qualifiersInAB(2);

        qualifiersInAB[0].isConjugate = conjugate ? 1 : 0;
        qualifiersInAB[0].isConstant = 1;
        qualifiersInAB[0].requiresGradient = 0;

        qualifiersInAB[1].isConjugate = conjugate ? 1 : 0;
        qualifiersInAB[1].isConstant = 1;
        qualifiersInAB[1].requiresGradient = 0;

        int64_t dimAwobond = static_cast<int64_t>(GetTensorSize(siteA, false)) / siteA_node.virtualExtents_[indexA];
        int64_t dimBwobond = static_cast<int64_t>(GetTensorSize(siteB, false)) / siteB_node.virtualExtents_[indexB];

        std::vector<int32_t> modesOutAwobond = GetTensorModes(siteA, false);
        std::vector<int32_t> modesOutBwobond = GetTensorModes(siteB, false);
        std::vector<int64_t> extentsOutAwobond = GetTensorExtents(siteA, false);
        std::vector<int64_t> extentsOutBwobond = GetTensorExtents(siteB, false);

        size_t bond_indexA = siteA_node.physModes_.size() + indexA;
        size_t bond_indexB = siteB_node.physModes_.size() + indexB;

        modesOutAwobond.erase(modesOutAwobond.begin() + bond_indexA);
        extentsOutAwobond.erase(extentsOutAwobond.begin() + bond_indexA);
        modesOutBwobond.erase(modesOutBwobond.begin() + bond_indexB);
        extentsOutBwobond.erase(extentsOutBwobond.begin() + bond_indexB);

        std::vector<int32_t> modesOutAB = modesOutAwobond;
        modesOutAB.insert(modesOutAB.end(), modesOutBwobond.begin(), modesOutBwobond.end());

        std::vector<int64_t> extentsOutAB = extentsOutAwobond;
        extentsOutAB.insert(extentsOutAB.end(), extentsOutBwobond.begin(), extentsOutBwobond.end());

        size_t dimOutAB = static_cast<size_t>(dimAwobond) * static_cast<size_t>(dimBwobond);

        void* tensorOutAB;
        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutAB, dimOutAB * sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesInAB,
                                            extentsInAB,
                                            qualifiersInAB,
                                            tensorsInAB,
                                            modesOutAB,
                                            dimOutAB,
                                            tensorOutAB,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {{0, 1}},
                                            workSpacePreference_);

        return tensorOutAB;
    }

    void TensorNetwork::SetTwoSiteVector(size_t siteA, 
                                         size_t siteB,
                                         const void* tensorInAB,
                                         int64_t maxVirtualExtent,
                                         size_t thread_num,
                                         bool verbose)
    {
        if(check_)
        {
            if((siteB >= numSites_) || (siteA >= numSites_))
            {
                throw std::invalid_argument("TensorNetwork::SetTwoSiteVector: "
                                            "Site index can not exceed maximal number of sites.");
            }
        }

        auto& siteA_node = nodes_[siteA];
        auto& siteB_node = nodes_[siteB];

        auto itA = siteA_node.neighbors_.find(siteB);
        auto itB = siteB_node.neighbors_.find(siteA);
        
        if(check_)
        {
            if((itA == siteA_node.neighbors_.end()) || (itB == siteB_node.neighbors_.end()))
            {
                throw std::invalid_argument("TensorNetwork::SetTwoSiteVector: "
                                             "Site " + std::to_string(siteA) + " must be a neighbor of site " 
                                             + std::to_string(siteB) + ".");
            }

            if(!siteA_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::SetTwoSiteVector: "
                                         "Duplicate bonds should be excluded for site " 
                                         + std::to_string(siteA) + ".");
            }

            if(!siteB_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::SetTwoSiteVector: "
                                         "Duplicate bonds should be excluded for site " 
                                         + std::to_string(siteB) + ".");
            }
        }

        size_t indexA = itA->second;
        size_t indexB = itB->second;

        size_t bond_indexA = siteA_node.physModes_.size() + indexA;
        size_t bond_indexB = siteB_node.physModes_.size() + indexB;

        std::vector<int32_t> modesInAwobond = GetTensorModes(siteA, false);
        std::vector<int32_t> modesInBwobond = GetTensorModes(siteB, false);
        std::vector<int64_t> extentsInAwobond = GetTensorExtents(siteA, false);
        std::vector<int64_t> extentsInBwobond = GetTensorExtents(siteB, false);

        modesInAwobond.erase(modesInAwobond.begin() + bond_indexA);
        extentsInAwobond.erase(extentsInAwobond.begin() + bond_indexA);
        modesInBwobond.erase(modesInBwobond.begin() + bond_indexB);
        extentsInBwobond.erase(extentsInBwobond.begin() + bond_indexB);

        std::vector<int32_t> modesInAB = modesInAwobond;
        modesInAB.insert(modesInAB.end(), modesInBwobond.begin(), modesInBwobond.end());

        std::vector<int64_t> extentsInAB = extentsInAwobond;
        extentsInAB.insert(extentsInAB.end(), extentsInBwobond.begin(), extentsInBwobond.end());

        int64_t dimAwobond = static_cast<int64_t>(GetTensorSize(siteA, false)) / siteA_node.virtualExtents_[indexA];
        int64_t dimBwobond = static_cast<int64_t>(GetTensorSize(siteB, false)) / siteB_node.virtualExtents_[indexB];
        
        int64_t extentABbond = (maxVirtualExtent > 0L) ? std::min({dimAwobond, dimBwobond, std::min({maxVirtualExtent_, maxVirtualExtent})}) : 
                                                         std::min({dimAwobond, dimBwobond, maxVirtualExtent_});

        siteA_node.virtualExtents_[indexA] = extentABbond;
        siteB_node.virtualExtents_[indexB] = extentABbond;

        std::vector<int32_t> modesOutA = GetTensorModes(siteA, false);
        std::vector<int64_t> extentsOutA = GetTensorExtents(siteA, false);

        std::vector<int32_t> modesOutB = GetTensorModes(siteB, false);
        std::vector<int64_t> extentsOutB = GetTensorExtents(siteB, false);

        void* tensorOutA;
        void* tensorOutB;

        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutA, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemset(tensorOutA, 0, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutB, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemset(tensorOutB, 0, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

        int64_t newABbondExtent = 0;

        CuTensorNetMethods::ApplyTensorSVD(handle_.at(thread_num),
                                           streams_.at(thread_num),
                                           svdConfig_.at(thread_num),
                                           modesInAB,
                                           extentsInAB,
                                           modesOutA,
                                           extentsOutA,
                                           modesOutB,
                                           extentsOutB,
                                           {bond_indexA, bond_indexB},
                                           tensorInAB,
                                           tensorOutA,
                                           tensorOutB,
                                           newABbondExtent,
                                           workSpacePreference_,
                                           verbose);

        siteA_node.virtualExtents_[indexA] = newABbondExtent;
        siteB_node.virtualExtents_[indexB] = newABbondExtent;
        
        HANDLE_CUDA_ERROR(cudaFree(tensors_[siteA]));
        HANDLE_CUDA_ERROR(cudaFree(tensors_[siteB]));

        tensors_[siteA] = tensorOutA;
        tensors_[siteB] = tensorOutB; 
    }

    std::set<size_t> TensorNetwork::OrthogonalizeAround(size_t center_id, size_t pre_center_id)
    {
        if(check_)
        {
            if(center_id >= numSites_)
            {
                throw std::invalid_argument("TensorNetwork::OrthogonalizeAround: "
                                             "Site index can not exceed maximal number of sites.");
            }

            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::OrthogonalizeAround: "
                                         "The state is not initialized.");
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                if(!nodes_[i].extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::OrthogonalizeAround: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        int32_t left_bond_mode = nextMode_;

        bool loop_free;

        graphTraversalType traversal = GetGraphTraversalToRoot(graph_, numSites_, center_id, loop_free);

        if(loop_free)
        {            
            if((pre_center_id != std::numeric_limits<size_t>::max()) && (pre_center_id != center_id))
            {
                std::vector<size_t> parentMap(numSites_, std::numeric_limits<size_t>::max());
            
                for(const auto& edge : traversal) 
                {
                    parentMap[edge.first] = edge.second;
                }
            
                graphTraversalType short_traversal;
                size_t current = pre_center_id;
            
                while(current != center_id) 
                {
                    size_t parent = parentMap[current];

                    if(parent == std::numeric_limits<size_t>::max()) 
                    {
                        throw std::runtime_error("TensorNetwork::OrthogonalizeAround: "
                                                 "No path from pre_center_id to center_id.");
                    }

                    short_traversal.emplace_back(current, parent);

                    current = parent;
                }

                traversal = short_traversal;
            }
            else if(pre_center_id == center_id)
            {
                return std::set<size_t>{};
            }
            
            for(const auto& [siteA, siteB] : traversal)
            {
                auto& siteA_node = nodes_[siteA];
                auto& siteB_node = nodes_[siteB];

                auto itA = siteA_node.neighbors_.find(siteB);
                auto itB = siteB_node.neighbors_.find(siteA);

                size_t indexA = itA->second;
                size_t indexB = itB->second;

                size_t bond_indexA = siteA_node.physModes_.size() + indexA;
                size_t bond_indexB = siteB_node.physModes_.size() + indexB;

                std::vector<int32_t> modesInA = GetTensorModes(siteA, false);
                std::vector<int64_t> extentsInA = GetTensorExtents(siteA, false);

                int32_t right_bond_mode = modesInA[bond_indexA];
                int64_t old_extentABbond = extentsInA[bond_indexA];

                std::vector<int32_t> modesOutA = modesInA;
                std::vector<int64_t> extentsOutA = extentsInA;

                int64_t dimAwobond = static_cast<int64_t>(GetTensorSize(siteA, false)) / siteA_node.virtualExtents_[indexA];

                int64_t new_extentABbond = std::min({dimAwobond, old_extentABbond});

                modesOutA[bond_indexA] = left_bond_mode;
                extentsOutA[bond_indexA] = new_extentABbond;

                std::vector<int32_t> modesOutAr = {left_bond_mode, right_bond_mode};
                std::vector<int64_t> extentsOutAr = {new_extentABbond, old_extentABbond};

                void* tensorOutA;
                void* tensorOutAr;

                HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutA, static_cast<size_t>(dimAwobond) * static_cast<size_t>(new_extentABbond) * sizeof(complexType)));
                HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutAr, static_cast<size_t>(new_extentABbond) * static_cast<size_t>(old_extentABbond) * sizeof(complexType)));

                CuTensorNetMethods::ApplyTensorQR(handle_[0],
                                                  streams_[0],
                                                  modesInA,
                                                  extentsInA,
                                                  modesOutA,
                                                  extentsOutA,
                                                  modesOutAr,
                                                  extentsOutAr,
                                                  tensors_[siteA],
                                                  tensorOutA,
                                                  tensorOutAr);

                siteA_node.virtualExtents_[indexA] = new_extentABbond;

                HANDLE_CUDA_ERROR(cudaFree(tensors_[siteA]));

                tensors_[siteA] = tensorOutA;

                std::vector<std::vector<int32_t>> modesInArB(2);
                std::vector<std::vector<int64_t>> extentsInArB(2);

                modesInArB[0] = modesOutAr;
                extentsInArB[0] = extentsOutAr;

                modesInArB[1] = GetTensorModes(siteB, false);
                extentsInArB[1] = GetTensorExtents(siteB, false);

                std::vector<const void*> tensorsInArB(2);

                tensorsInArB[0] = tensorOutAr;
                tensorsInArB[1] = tensors_[siteB];

                std::vector<cutensornetTensorQualifiers_t> qualifiersInArB(2);

                qualifiersInArB[0].isConjugate = 0;
                qualifiersInArB[0].isConstant = 1;
                qualifiersInArB[0].requiresGradient = 0;

                qualifiersInArB[1].isConjugate = 0;
                qualifiersInArB[1].isConstant = 1;
                qualifiersInArB[1].requiresGradient = 0;

                std::vector<int32_t> modesOutB = modesInArB[1];

                modesOutB[bond_indexB] = left_bond_mode;

                siteB_node.virtualExtents_[indexB] = new_extentABbond;

                int64_t dimOutB = static_cast<int64_t>(GetTensorSize(siteB, false));

                void* tensorOutB;
                HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutB, dimOutB * sizeof(complexType)));

                CuTensorNetMethods::ContractTensors(handle_[0],
                                                    streams_[0],
                                                    modesInArB,
                                                    extentsInArB,
                                                    qualifiersInArB,
                                                    tensorsInArB,
                                                    modesOutB,
                                                    dimOutB,
                                                    tensorOutB,
                                                    0.8 / static_cast<double>(numStreams_),
                                                    workSpaceLimit_,
                                                    {{0, 1}},
                                                    workSpacePreference_); 

                HANDLE_CUDA_ERROR(cudaFree(tensorOutAr));
                HANDLE_CUDA_ERROR(cudaFree(tensors_[siteB]));

                tensors_[siteB] = tensorOutB;
            }

            std::set<size_t> visited_nodes;
            
            for(const auto& p : traversal) 
            {
                visited_nodes.insert(p.first);
                visited_nodes.insert(p.second);
            }

            return visited_nodes;
        }
        else
        {
            return std::set<size_t>{};
        }
    }

    void TensorNetwork::Shrink(bool verbose, size_t thread_num)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::Shrink: "
                                         "The state is not initialized.");
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                if(!nodes_[i].extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::Shrink: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        for(const auto& [siteA, siteB] : graphTraversalToRoot_)
        {
            void* tensorAB = ComputeTwoSiteVector(siteA, siteB, false, thread_num);

            SetTwoSiteVector(siteA, 
                             siteB, 
                             tensorAB, 
                             0UL, 
                             thread_num, 
                             verbose);

            HANDLE_CUDA_ERROR(cudaFree(tensorAB));
        }
    }

    void TensorNetwork::ApplySingleSiteGate(size_t site, 
                                            const void* operatorData,
                                            std::vector<int32_t> operatorModes,
                                            std::vector<int64_t> operatorExtents,
                                            size_t thread_num)
    {
        if(check_)
        {
            if(site >= numSites_)
            {
                throw std::invalid_argument("TensorNetwork::ApplySingleSiteGate: "
                                             "Site index can not exceed maximal number of sites.");
            }

            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplySingleSiteGate: "
                                         "The state is not initialized.");
            }

            std::unordered_map<int32_t, size_t> freq;
            for(const auto& opm : operatorModes) 
            {
                if(++freq[opm] > 2UL) 
                {
                    throw std::runtime_error("TensorNetwork::ApplySingleSiteGate: "
                                             "The operator has modes that repeat more than twice.");
                }
            }
        }

        int32_t currentMode = nextMode_;

        size_t operatorNumModes = operatorModes.size();
        
        auto maxOpMode = std::max_element(operatorModes.begin(), operatorModes.end());

        if(maxOpMode != operatorModes.cend())
        {
            int32_t currentOpMode = (*maxOpMode) + 1;

            if(currentMode < currentOpMode)
            {
                currentMode = currentOpMode;
            }
        }

        auto& site_node = nodes_[site];

        std::vector<std::vector<int32_t>> modesIn(2);
        std::vector<std::vector<int64_t>> extentsIn(2);

        modesIn[0] = GetTensorModes(site);
        extentsIn[0] = GetTensorExtents(site);

        std::vector<int32_t> physModesOutSite = site_node.physModes_;
        std::vector<bool> where_new_index(operatorNumModes, true);

        size_t opSites = 0UL;

        for(size_t i = 0UL; i < physModesOutSite.size(); ++i) 
        {
            auto& mode = physModesOutSite[i];

            size_t enp = 0UL;
            for(size_t j = 0UL; j < operatorNumModes; ++j)
            {
                auto& opm = operatorModes[j];

                if(opm == mode)
                {
                    ++enp;

                    if(check_)
                    {
                        if(site_node.physExtents_[i] != operatorExtents[j])
                        {
                            throw std::runtime_error("TensorNetwork::ApplySingleSiteGate: "
                                                     "The operator extent and the corresponding extent of site " 
                                                     + std::to_string(site) + " do not match.");
                        }

                        ++opSites;
                    }
                    
                    if(where_new_index[j])
                    {
                        where_new_index[j] = false;
                    }

                    if(enp == 2UL)
                    {
                        mode = currentMode++;
                        opm = mode;

                        break;
                    }
                }
            }
        }

        if(check_)
        {
            if(opSites == 0UL)
            {
                throw std::runtime_error("TensorNetwork::ApplySingleSiteGate: "
                                         "The gate doesn't have any common modes with site " 
                                          + std::to_string(site) + ".");
            }
        }

        for(size_t i = 0UL; i < operatorNumModes; ++i)
        {
            if(where_new_index[i])
            {
                site_node.extra_virtualModes_.push_back(operatorModes[i]);
                site_node.extra_virtualExtents_.push_back(operatorExtents[i]);
            }
        }

        modesIn[1] = operatorModes;
        extentsIn[1] = operatorExtents;

        std::vector<const void*> tensorsIn(2);
        
        tensorsIn[0] = tensors_[site];
        tensorsIn[1] = operatorData;

        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(2);

        qualifiersIn[0].isConjugate = 0;
        qualifiersIn[0].isConstant = 1;
        qualifiersIn[0].requiresGradient = 0;

        qualifiersIn[1].isConjugate = 0;
        qualifiersIn[1].isConstant = 1;
        qualifiersIn[1].requiresGradient = 0;

        std::vector<int32_t> modesOutSite = physModesOutSite;
        modesOutSite.insert(modesOutSite.end(), site_node.virtualModes_.begin(), site_node.virtualModes_.end());
        modesOutSite.insert(modesOutSite.end(), site_node.extra_virtualModes_.begin(), site_node.extra_virtualModes_.end());

        size_t dimOut = GetTensorSize(site);

        void* tensorOut;
        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOut, dimOut * sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesIn,
                                            extentsIn,
                                            qualifiersIn,
                                            tensorsIn,
                                            modesOutSite,
                                            dimOut,
                                            tensorOut,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {{0, 1}},
                                            workSpacePreference_);

        HANDLE_CUDA_ERROR(cudaFree(tensors_[site]));
        tensors_[site] = tensorOut;
    }

    void TensorNetwork::ApplyTwoSiteGate(size_t siteA, 
                                         size_t siteB, 
                                         const void* operatorData,
                                         std::vector<int32_t> operatorModes,
                                         std::vector<int64_t> operatorExtents,
                                         size_t thread_num,
                                         bool verbose)
    {     
        if(check_)
        {
            if((siteB >= numSites_) || (siteA >= numSites_))
            {
                throw std::invalid_argument("TensorNetwork::ApplyTwoSiteGate: "
                                            "Site index can not exceed maximal number of sites.");
            }

            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplyTwoSiteGate: "
                                         "The state is not initialized.");
            }

            std::unordered_map<int32_t, size_t> freq;
            for(const auto& opm : operatorModes) 
            {
                ++freq[opm];
            }

            for(const auto& [key, val] : freq)
            {
                if(val != 2UL) 
                {
                    throw std::runtime_error("TensorNetwork::ApplySingleSiteGate: "
                                             "The operator has modes that are not exactly repeated twice.");
                }
            }
        }

        auto& siteA_node = nodes_[siteA];
        auto& siteB_node = nodes_[siteB];

        auto itA = siteA_node.neighbors_.find(siteB);
        auto itB = siteB_node.neighbors_.find(siteA);
        
        if(check_)
        {
            if((itA == siteA_node.neighbors_.end()) || (itB == siteB_node.neighbors_.end()))
            {
                throw std::invalid_argument("TensorNetwork::ApplyTwoSiteGate: "
                                             "Site " + std::to_string(siteA) + " must be a neighbor of site " 
                                             + std::to_string(siteB) + ".");
            }

            if(!siteA_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::ApplyTwoSiteGate: "
                                         "Duplicate bonds should be excluded for site " 
                                         + std::to_string(siteA) + ".");
            }

            if(!siteB_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::ApplyTwoSiteGate: "
                                         "Duplicate bonds should be excluded for site " 
                                         + std::to_string(siteB) + ".");
            }
        }

        size_t indexA = itA->second;
        size_t indexB = itB->second;
        
        int32_t currentMode = nextMode_;

        std::vector<std::vector<int32_t>> modesInAB(3);
        std::vector<std::vector<int64_t>> extentsInAB(3);
        
        modesInAB[0] = GetTensorModes(siteA, false);
        extentsInAB[0] = GetTensorExtents(siteA, false);

        modesInAB[1] = GetTensorModes(siteB, false);
        extentsInAB[1] = GetTensorExtents(siteB, false);

        size_t operatorNumModes = operatorModes.size();
        
        std::vector<int32_t> physModesOutA = siteA_node.physModes_;
        std::vector<int32_t> physModesOutB = siteB_node.physModes_;

        size_t opSites = 0UL;

        for(size_t i = 0UL; i < physModesOutA.size(); ++i)
        {
            auto& mode = physModesOutA[i];

            size_t enp = 0UL;
            for(size_t j = 0UL; j < operatorNumModes; ++j)
            {
                auto& opm = operatorModes[j];

                if(opm == mode)
                {
                    ++enp;
                    
                    if(check_)
                    {
                        if(siteA_node.physExtents_[i] != operatorExtents[j])
                        {
                            throw std::runtime_error("TensorNetwork::ApplyTwoSiteGate: "
                                                     "The operator extent and the corresponding extent of site " 
                                                     + std::to_string(siteA) + " do not match.");
                        }

                        ++opSites;
                    }

                    if(enp == 2UL)
                    {
                        mode = currentMode++;
                        opm = mode;

                        break;
                    }
                }
            }
        }

        for(size_t i = 0UL; i < physModesOutB.size(); ++i)
        {
            auto& mode = physModesOutB[i];

            size_t enp = 0UL;
            for(size_t j = 0UL; j < operatorNumModes; ++j)
            {
                auto& opm = operatorModes[j];

                if(opm == mode)
                {
                    ++enp;

                    if(check_)
                    {
                        if(siteB_node.physExtents_[i] != operatorExtents[j])
                        {
                            throw std::runtime_error("TensorNetwork::ApplyTwoSiteGate: "
                                                     "The operator extent and the corresponding extent of site " 
                                                     + std::to_string(siteB) + " do not match.");
                        }

                        ++opSites;
                    }
                    
                    if(enp == 2UL)
                    {
                        mode = currentMode++;
                        opm = mode;

                        break;
                    }
                }
            }
        }

        if(check_)
        {    
            if(opSites != operatorNumModes)
            {
                throw std::runtime_error("TensorNetwork::ApplyTwoSiteGate: "
                                         "The operator modes and the site modes do not match.");
            }
        }

        modesInAB[2] = operatorModes;
        extentsInAB[2] = operatorExtents;

        std::vector<const void*> tensorsInAB(3);
        
        tensorsInAB[0] = tensors_[siteA];
        tensorsInAB[1] = tensors_[siteB];
        tensorsInAB[2] = operatorData;

        std::vector<cutensornetTensorQualifiers_t> qualifiersInAB(3);

        qualifiersInAB[0].isConjugate = 0;
        qualifiersInAB[0].isConstant = 1;
        qualifiersInAB[0].requiresGradient = 0;

        qualifiersInAB[1].isConjugate = 0;
        qualifiersInAB[1].isConstant = 1;
        qualifiersInAB[1].requiresGradient = 0;

        qualifiersInAB[2].isConjugate = 0;
        qualifiersInAB[2].isConstant = 1;
        qualifiersInAB[2].requiresGradient = 0;

        int64_t dimAwobond = static_cast<int64_t>(GetTensorSize(siteA, false)) / siteA_node.virtualExtents_[indexA];
        int64_t dimBwobond = static_cast<int64_t>(GetTensorSize(siteB, false)) / siteB_node.virtualExtents_[indexB];

        int64_t extentABbond = std::min({dimAwobond, dimBwobond, maxVirtualExtent_});

        siteA_node.virtualExtents_[indexA] = extentABbond;
        siteB_node.virtualExtents_[indexB] = extentABbond;

        std::vector<int32_t> modesOutA = physModesOutA;
        modesOutA.insert(modesOutA.end(), siteA_node.virtualModes_.begin(), siteA_node.virtualModes_.end());
        std::vector<int64_t> extentsOutA = GetTensorExtents(siteA, false);

        std::vector<int32_t> modesOutB = physModesOutB;
        modesOutB.insert(modesOutB.end(), siteB_node.virtualModes_.begin(), siteB_node.virtualModes_.end());
        std::vector<int64_t> extentsOutB = GetTensorExtents(siteB, false);

        std::vector<int32_t> modesOutAwobond = modesOutA;
        std::vector<int32_t> modesOutBwobond = modesOutB;
        std::vector<int64_t> extentsOutAwobond = extentsOutA;
        std::vector<int64_t> extentsOutBwobond = extentsOutB;

        size_t bond_indexA = siteA_node.physModes_.size() + indexA;
        size_t bond_indexB = siteB_node.physModes_.size() + indexB;

        modesOutAwobond.erase(modesOutAwobond.begin() + bond_indexA);
        extentsOutAwobond.erase(extentsOutAwobond.begin() + bond_indexA);
        modesOutBwobond.erase(modesOutBwobond.begin() + bond_indexB);
        extentsOutBwobond.erase(extentsOutBwobond.begin() + bond_indexB);

        std::vector<int32_t> modesOutAB = modesOutAwobond;
        modesOutAB.insert(modesOutAB.end(), modesOutBwobond.begin(), modesOutBwobond.end());

        std::vector<int64_t> extentsOutAB = extentsOutAwobond;
        extentsOutAB.insert(extentsOutAB.end(), extentsOutBwobond.begin(), extentsOutBwobond.end());

        size_t dimOutAB = static_cast<size_t>(dimAwobond) * static_cast<size_t>(dimBwobond);

        void* tensorOutAB;
        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutAB, dimOutAB * sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesInAB,
                                            extentsInAB,
                                            qualifiersInAB,
                                            tensorsInAB,
                                            modesOutAB,
                                            dimOutAB,
                                            tensorOutAB,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {{0, 1}, {0, 1}},
                                            workSpacePreference_);
        
        void* tensorOutA;
        void* tensorOutB;

        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutA, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemset(tensorOutA, 0, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutB, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemset(tensorOutB, 0, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

        int64_t newABbondExtent = 0;

        CuTensorNetMethods::ApplyTensorSVD(handle_.at(thread_num),
                                           streams_.at(thread_num),
                                           svdConfig_.at(thread_num),
                                           modesOutAB,
                                           extentsOutAB,
                                           modesOutA,
                                           extentsOutA,
                                           modesOutB,
                                           extentsOutB,
                                           {bond_indexA, bond_indexB},
                                           tensorOutAB,
                                           tensorOutA,
                                           tensorOutB,
                                           newABbondExtent,
                                           workSpacePreference_,
                                           verbose);

        siteA_node.virtualExtents_[indexA] = newABbondExtent;
        siteB_node.virtualExtents_[indexB] = newABbondExtent;

        HANDLE_CUDA_ERROR(cudaFree(tensorOutAB));
        
        HANDLE_CUDA_ERROR(cudaFree(tensors_[siteA]));
        HANDLE_CUDA_ERROR(cudaFree(tensors_[siteB]));

        tensors_[siteA] = tensorOutA;
        tensors_[siteB] = tensorOutB;
    }

    __global__ void ConjugateKernel(cuDoubleComplex* tensor_data, const size_t size) 
    {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size) return;

        cuDoubleComplex z = tensor_data[idx];

        tensor_data[idx] = make_cuDoubleComplex(z.x, -z.y);
    }

    TensorNetwork& TensorNetwork::Conjugate()
    {
        if(check_)
        { 
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::Conjugate: "
                                         "The state of the current TN is not initialized.");
            }
        }
        
        std::vector<std::thread> threads;

        auto thread_func = [this](size_t thread_num, size_t start, size_t end) 
        {
            for(size_t i = start; i < end; ++i)
            {                
                cuDoubleComplex* tensor_data = static_cast<cuDoubleComplex*>(this->tensors_[i]);

                size_t dim = this->GetTensorSize(i);

                int64_t blockSize = 256L;
                int64_t numBlocks = (static_cast<int64_t>(dim) + blockSize - 1L) / blockSize;
                
                ConjugateKernel <<< numBlocks, blockSize, 0, streams_.at(thread_num) >>> (tensor_data, dim);
            }
        };

        size_t chunk_size = (numSites_ + numStreams_ - 1) / numStreams_;

        for(size_t th = 0; th < numStreams_; ++th) 
        {
            size_t start = th * chunk_size;
            size_t end = std::min(start + chunk_size, numSites_);

            if (start < numSites_) 
            {
                threads.emplace_back(thread_func, th, start, end);
            } 
            else 
            {
                break;
            }
        }

        for(auto& thread : threads) 
        {
            thread.join();
        }

        SynchronizeStreams();

        threads.clear();

        return *this;
    }

    __global__ void PermuteKernel(const cuDoubleComplex* __restrict__ tensor_in,
                                  cuDoubleComplex* __restrict__ tensor_out,
                                  size_t sdim1, size_t sdim2, size_t total_size)
    {
        const size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(out_idx >= total_size) return;

        size_t c = out_idx / (sdim1 * sdim2);
        size_t rem = out_idx % (sdim1 * sdim2);
        size_t a = rem / sdim2;
        size_t b = rem % sdim2;

        size_t in_idx = a + sdim1 * (b + sdim2 * c);

        tensor_out[out_idx] = tensor_in[in_idx];
    }

    TensorNetwork& TensorNetwork::Transpose()
    {
        if(check_)
        { 
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::Transpose: "
                                         "The state of the current TN is not initialized.");
            }

            for(size_t i = 0; i < numSites_; ++i)
            {
                if(nodes_[i].physModes_.size() % 2UL != 0)
                {
                    throw std::runtime_error("TensorNetwork::Transpose: "
                                             "The current TN must be a operator.");
                }
            }
        }

        std::vector<std::thread> threads;

        auto thread_func = [this](size_t thread_num, size_t start, size_t end) 
        {
            for(size_t i = start; i < end; ++i)
            {                
                cuDoubleComplex* in_ten_accessor = static_cast<cuDoubleComplex*>(this->tensors_[i]);

                size_t total_size = this->GetTensorSize(i);

                size_t sdim1 = static_cast<size_t>(std::accumulate(this->nodes_[i].physExtents_.begin(),
                                                                   this->nodes_[i].physExtents_.begin() + (this->nodes_[i].physExtents_.size() / 2UL),
                                                                   1L,
                                                                   [](int64_t acc, const int64_t& current) {return acc * current;}));

                size_t sdim2 = static_cast<size_t>(std::accumulate(this->nodes_[i].physExtents_.begin() + (this->nodes_[i].physExtents_.size() / 2UL),
                                                                   this->nodes_[i].physExtents_.end(),
                                                                   1L,
                                                                   [](int64_t acc, const int64_t& current) {return acc * current;}));

                void* new_tensor;
                HANDLE_CUDA_ERROR(cudaMalloc(&new_tensor, total_size * sizeof(complexType)));

                cuDoubleComplex* out_ten_accessor = static_cast<cuDoubleComplex*>(new_tensor);

                int64_t blockSize = 256L;
                int64_t numBlocks = (static_cast<int64_t>(total_size) + blockSize - 1L) / blockSize;
                
                PermuteKernel <<< numBlocks, blockSize, 0, streams_.at(thread_num) >>> (in_ten_accessor, out_ten_accessor, sdim1, sdim2, total_size);

                this->SynchronizeStreams({thread_num});

                HANDLE_CUDA_ERROR(cudaFree(tensors_[i]));

                tensors_[i] = new_tensor;
            }
        };

        size_t chunk_size = (numSites_ + numStreams_ - 1) / numStreams_;

        for(size_t th = 0; th < numStreams_; ++th) 
        {
            size_t start = th * chunk_size;
            size_t end = std::min(start + chunk_size, numSites_);

            if (start < numSites_) 
            {
                threads.emplace_back(thread_func, th, start, end);
            } 
            else 
            {
                break;
            }
        }

        for(auto& thread : threads) 
        {
            thread.join();
        }

        threads.clear();

        return *this;
    }

    __global__ void MultiplyTensorByScalarKernel(cuDoubleComplex* tensor_data,
                                                 const cuDoubleComplex cuScalar,
                                                 const size_t size)
    {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size) return;
        
        tensor_data[idx] = cuCmul(tensor_data[idx], cuScalar);
    }

    void TensorNetwork::ApplyScalar(complexType scalar, const std::vector<size_t>& nodes)
    {
        if(check_)
        { 
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplyScalar: "
                                         "The state of the current TN is not initialized.");
            }
        }
        
        size_t num = nodes.empty() ? 1UL : nodes.size();
        
        scalar = std::pow(scalar, 1.0 / static_cast<double>(num));
        
        cuDoubleComplex cuScalar = make_cuDoubleComplex(scalar.real(), scalar.imag());

        std::vector<std::thread> threads;

        auto thread_func = [this, &cuScalar, &nodes](size_t thread_num, size_t start, size_t end) 
        {
            for(size_t k = start; k < end; ++k)
            {
                size_t i = nodes.empty() ? k : nodes[k];
                
                cuDoubleComplex* tensor_data = static_cast<cuDoubleComplex*>(this->tensors_[i]);

                size_t dim = this->GetTensorSize(i);

                int64_t blockSize = 256L;
                int64_t numBlocks = (static_cast<int64_t>(dim) + blockSize - 1L) / blockSize;
                
                MultiplyTensorByScalarKernel <<< numBlocks, blockSize, 0, streams_.at(thread_num) >>> (tensor_data, cuScalar, dim);
            }
        };

        size_t chunk_size = (num + numStreams_ - 1UL) / numStreams_;

        for(size_t th = 0; th < numStreams_; ++th) 
        {
            size_t start = th * chunk_size;
            size_t end = std::min(start + chunk_size, num);

            if (start < num) 
            {
                threads.emplace_back(thread_func, th, start, end);
            } 
            else 
            {
                break;
            }
        }

        for(auto& thread : threads) 
        {
            thread.join();
        }

        SynchronizeStreams();

        threads.clear();
    }

    TensorNetwork& TensorNetwork::operator*=(complexType scalar)
    {
        ApplyScalar(scalar);

        return *this;
    }
    
    TensorNetwork TensorNetwork::operator*(complexType scalar) const
    {
        TensorNetwork result(*this);

        result.ApplyScalar(scalar);

        return result;
    }

    TensorNetwork operator*(complexType scalar, const TensorNetwork& Psi)
    {
        return Psi * scalar;
    }

    void TensorNetwork::ApplyTensorNetOperator(const TensorNetwork* Omega, bool verbose)
    {
        if(check_)
        { 
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                         "The state of the current TN is not initialized.");
            }

            if(Omega->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                         "The state of Omega TN is not initialized.");
            }

            if(numSites_ != Omega->numSites_)
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                         "The number of sites of the current TN differs from that of Omega TN.");
            }

            if(graph_ != Omega->graph_)
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                         "The graphs of the current TN and the Omega TN being applied differ from each other.");
            }

            for(size_t i = 0; i < numSites_; ++i)
            {
                const auto& node_i = this->nodes_[i];
                const auto& Omega_node_i = Omega->nodes_[i];

                if(!node_i.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(!Omega_node_i.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of Omega TN.");
                }

                if(Omega_node_i.physModes_.size() % 2UL != 0)
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                             "The Omega TN must be a operator.");
                }

                if((node_i.physModes_.size() * 2UL != Omega_node_i.physModes_.size()) && (node_i.physModes_.size() != Omega_node_i.physModes_.size()))
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetOperator: "
                                             "The phys modes of the current TN and Omega TN are not suitable at site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        int32_t currentMode = nextMode_;

        auto applying_func = [this, Omega, currentMode](size_t i) 
        {
            const auto& node_i = this->nodes_[i];
            const auto& Omega_node_i = Omega->nodes_[i];

            std::vector<int32_t> Omega_modes_i;
            if(node_i.physModes_.size() * 2UL == Omega_node_i.physModes_.size())
            {
                Omega_modes_i.insert(Omega_modes_i.end(), node_i.physModes_.begin(), node_i.physModes_.end());
                Omega_modes_i.insert(Omega_modes_i.end(), node_i.physModes_.begin(), node_i.physModes_.end());
            }
            else
            {
                Omega_modes_i.insert(Omega_modes_i.end(), node_i.physModes_.begin() + (node_i.physModes_.size() / 2UL), node_i.physModes_.end());
                Omega_modes_i.insert(Omega_modes_i.end(), node_i.physModes_.begin() + (node_i.physModes_.size() / 2UL), node_i.physModes_.end());
            }

            std::vector<int32_t> Omega_virtualModes_i = Omega_node_i.virtualModes_;
            std::transform(Omega_virtualModes_i.begin(), Omega_virtualModes_i.end(), Omega_virtualModes_i.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
            Omega_modes_i.insert(Omega_modes_i.end(), Omega_virtualModes_i.begin(), Omega_virtualModes_i.end());

            std::vector<int64_t> Omega_extents_i = Omega->GetTensorExtents(i, false);
                
            this->ApplySingleSiteGate(i, Omega->tensors_[i], Omega_modes_i, Omega_extents_i, 0UL);
        };

        std::vector<bool> appied_nodes(numSites_, false);

        for(const auto& [siteA, siteB] : graphTraversalToRoot_)
        {
            if(!appied_nodes.at(siteA))
            {
                appied_nodes[siteA] = true;

                applying_func(siteA);
            }

            if(!appied_nodes.at(siteB))
            {
                appied_nodes[siteB] = true;

                applying_func(siteB);
            }

            ExcludeExtraBond(siteA, siteB, 0UL, verbose);
        }
    }

    TensorNetwork& TensorNetwork::operator*=(const TensorNetwork& Psi)
    {
        ApplyTensorNetOperator(&Psi);

        return *this;
    }

    TensorNetwork TensorNetwork::operator*(const TensorNetwork& Psi) const
    {
        TensorNetwork result(*this);

        result.ApplyTensorNetOperator(&Psi);

        return result;
    }

    __global__ void CombineTensorsKernel(const complexType* __restrict__ tensorA, const int64_t* __restrict__ extentsA, const int64_t* __restrict__ stridesA,
                                         const complexType* __restrict__ tensorB, const int64_t* __restrict__ stridesB,
                                         complexType* __restrict__ tensorC, const int64_t* __restrict__ stridesC, 
                                         const int32_t numModes, size_t virtualBegin, size_t totalElementsA, size_t totalElementsB)
    {        
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= totalElementsA + totalElementsB) return;

        if(idx < totalElementsA)
        {           
            int64_t temp = static_cast<int64_t>(idx);

            int64_t indexC = 0;

            for(int32_t i = numModes - 1; i >= 0; --i) 
            {
                int64_t coord = temp / stridesA[i];

                temp %= stridesA[i];

                indexC += coord * stridesC[i];
            }

            tensorC[indexC] = tensorA[idx];
        }

        if(idx >= totalElementsA)
        {
            size_t b_idx = idx - totalElementsA;
            
            int64_t temp = static_cast<int64_t>(b_idx);

            int64_t indexC = 0;

            for (int32_t i = numModes - 1; i >= 0; --i) 
            {
                int64_t coord = temp / stridesB[i];

                temp %= stridesB[i];

                if(i >= virtualBegin)
                {
                    indexC += (coord + extentsA[i]) * stridesC[i];
                }
                else
                {
                    indexC += coord * stridesC[i];
                }
            }

            tensorC[indexC] = tensorB[b_idx];
        }
    }

    void TensorNetwork::AddTensorNet(const TensorNetwork* Psi, bool verbose)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                         "The state of the current TN is not initialized.");
            }

            if(Psi->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                         "The state of Psi TN is not initialized.");
            }

            if(graph_ != Psi->graph_)
            {
                throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                         "The graphs of the current TN and the Psi TN being added differ from each other.");
            }

            if(numSites_ != Psi->numSites_)
            {
                throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                         "The number of sites of the current TN differs from that of Psi TN.");
            }

            for(size_t i = 0; i < numSites_; ++i)
            {
                auto& siteI_node = this->nodes_[i];
                const auto& Psi_node = Psi->nodes_[i];

                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(!Psi_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of Psi TN.");
                }

                if(siteI_node.physModes_.size() != Psi_node.physModes_.size())
                {
                    throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                             "Physical mode number of site " + std::to_string(i) + 
                                             " of the current TN differs from that of Psi TN site.");
                }

                if(!std::equal(siteI_node.physExtents_.begin(), siteI_node.physExtents_.end(), Psi_node.physExtents_.begin()))
                {
                    throw std::runtime_error("TensorNetwork::AddTensorNet: "
                                             "Physical extents of site " + std::to_string(i) + 
                                             " of the current TN differs from those of Psi TN site.");
                }
            }
        }

        std::vector<void*> local_tensors(numSites_, nullptr);
        
        auto extending_func = [this, Psi, &local_tensors](size_t i) 
        {                    
            std::vector<int64_t> extentsA = this->GetTensorExtents(i, false);
            std::vector<int64_t> extentsB = Psi->GetTensorExtents(i, false);

            size_t numModes = extentsA.size();

            size_t virtualBegin = this->nodes_[i].physExtents_.size();

            std::vector<int64_t> extentsC(numModes);

            for(size_t j = 0; j < numModes; ++j)
            {
                if(virtualBegin <= j)
                {
                    extentsC[j] = extentsA[j] + extentsB[j];
                }
                else
                {
                    extentsC[j] = extentsA[j];
                }
            }

            size_t dimA = this->GetTensorSize(i, false);
            size_t dimB = Psi->GetTensorSize(i, false);

            size_t dimC = static_cast<size_t>(std::accumulate(extentsC.begin(),
                                              extentsC.end(),
                                              1L,
                                              [](int64_t acc, const int64_t& current) {return acc * current;}));

            HANDLE_CUDA_ERROR(cudaMalloc(&(local_tensors[i]), dimC * sizeof(complexType)));
            HANDLE_CUDA_ERROR(cudaMemset(local_tensors[i], 0, dimC * sizeof(complexType)));

            std::vector<int64_t> stridesA(numModes, 1);
            std::vector<int64_t> stridesB(numModes, 1);
            std::vector<int64_t> stridesC(numModes, 1);

            for(size_t j = 1; j < numModes; ++j) 
            {
                stridesA[j] = stridesA[j - 1] * extentsA[j - 1];
                stridesB[j] = stridesB[j - 1] * extentsB[j - 1];
                stridesC[j] = stridesC[j - 1] * extentsC[j - 1];
            }

            int64_t* extentsA_device;
            int64_t* stridesA_device;
            int64_t* stridesB_device;
            int64_t* stridesC_device;

            HANDLE_CUDA_ERROR(cudaMalloc((void**)(&extentsA_device), numModes * sizeof(int64_t)));
            HANDLE_CUDA_ERROR(cudaMemcpy((void*)extentsA_device, (void*)(extentsA.data()), numModes * sizeof(int64_t), cudaMemcpyHostToDevice));

            HANDLE_CUDA_ERROR(cudaMalloc((void**)(&stridesA_device), numModes * sizeof(int64_t)));
            HANDLE_CUDA_ERROR(cudaMemcpy((void*)stridesA_device, (void*)(stridesA.data()), numModes * sizeof(int64_t), cudaMemcpyHostToDevice));

            HANDLE_CUDA_ERROR(cudaMalloc((void**)(&stridesB_device), numModes * sizeof(int64_t)));
            HANDLE_CUDA_ERROR(cudaMemcpy((void*)stridesB_device, (void*)(stridesB.data()), numModes * sizeof(int64_t), cudaMemcpyHostToDevice));

            HANDLE_CUDA_ERROR(cudaMalloc((void**)(&stridesC_device), numModes * sizeof(int64_t)));
            HANDLE_CUDA_ERROR(cudaMemcpy((void*)stridesC_device, (void*)(stridesC.data()), numModes * sizeof(int64_t), cudaMemcpyHostToDevice));

            int64_t blockSize = 256;
            int64_t numBlocks = (static_cast<int64_t>(dimA) + static_cast<int64_t>(dimB) + blockSize - 1L) / blockSize;

            const complexType* tensorA = static_cast<const complexType*>(this->tensors_[i]);
            const complexType* tensorB = static_cast<const complexType*>(Psi->tensors_[i]);
            complexType* tensorC = static_cast<complexType*>(local_tensors[i]);

            CombineTensorsKernel <<< numBlocks, blockSize, 0, streams_[0] >>> (tensorA, 
                                                                               extentsA_device,
                                                                               stridesA_device,
                                                                               tensorB, 
                                                                               stridesB_device, 
                                                                               tensorC,
                                                                               stridesC_device, 
                                                                               static_cast<int32_t>(numModes),
                                                                               virtualBegin, dimA, dimB);
                
            SynchronizeStreams(std::vector<size_t>{0});

            HANDLE_CUDA_ERROR(cudaFree(extentsA_device));
            HANDLE_CUDA_ERROR(cudaFree(stridesA_device));
            HANDLE_CUDA_ERROR(cudaFree(stridesB_device));
            HANDLE_CUDA_ERROR(cudaFree(stridesC_device));

            auto& virtualExtentsI = this->nodes_[i].virtualExtents_;

            for(size_t j = virtualBegin; j < numModes; ++j)
            {
                virtualExtentsI[j - virtualBegin] = extentsC[j];
            }
        };

        for(const auto& [siteA, siteB] : graphTraversalToRoot_)
        {
            if(local_tensors[siteA] == nullptr)
            {
                extending_func(siteA);
            }

            if(local_tensors[siteB] == nullptr)
            {
                extending_func(siteB);
            }

            auto& siteA_node = nodes_[siteA];
            auto& siteB_node = nodes_[siteB];

            auto itA = siteA_node.neighbors_.find(siteB);
            auto itB = siteB_node.neighbors_.find(siteA);

            size_t indexA = itA->second;
            size_t indexB = itB->second;

            std::vector<std::vector<int32_t>> modesInAB(2);
            std::vector<std::vector<int64_t>> extentsInAB(2);
                
            modesInAB[0] = GetTensorModes(siteA, false);
            extentsInAB[0] = GetTensorExtents(siteA, false);

            modesInAB[1] = GetTensorModes(siteB, false);
            extentsInAB[1] = GetTensorExtents(siteB, false);

            std::vector<const void*> tensorsInAB(2);
                
            tensorsInAB[0] = local_tensors[siteA];
            tensorsInAB[1] = local_tensors[siteB];

            std::vector<cutensornetTensorQualifiers_t> qualifiersInAB(2);

            qualifiersInAB[0].isConjugate = 0;
            qualifiersInAB[0].isConstant = 1;
            qualifiersInAB[0].requiresGradient = 0;

            qualifiersInAB[1].isConjugate = 0;
            qualifiersInAB[1].isConstant = 1;
            qualifiersInAB[1].requiresGradient = 0;

            int64_t dimAwobond = static_cast<int64_t>(GetTensorSize(siteA, false)) / siteA_node.virtualExtents_[indexA];
            int64_t dimBwobond = static_cast<int64_t>(GetTensorSize(siteB, false)) / siteB_node.virtualExtents_[indexB];

            int64_t extentABbond = std::min({dimAwobond, dimBwobond, maxVirtualExtent_});

            siteA_node.virtualExtents_[indexA] = extentABbond;
            siteB_node.virtualExtents_[indexB] = extentABbond;

            std::vector<int32_t> modesOutA = GetTensorModes(siteA, false);
            std::vector<int64_t> extentsOutA = GetTensorExtents(siteA, false);

            std::vector<int32_t> modesOutB = GetTensorModes(siteB, false);
            std::vector<int64_t> extentsOutB = GetTensorExtents(siteB, false);

            std::vector<int32_t> modesOutAwobond = modesOutA;
            std::vector<int32_t> modesOutBwobond = modesOutB;
            std::vector<int64_t> extentsOutAwobond = extentsOutA;
            std::vector<int64_t> extentsOutBwobond = extentsOutB;

            size_t bond_indexA = siteA_node.physModes_.size() + indexA;
            size_t bond_indexB = siteB_node.physModes_.size() + indexB;

            modesOutAwobond.erase(modesOutAwobond.begin() + bond_indexA);
            extentsOutAwobond.erase(extentsOutAwobond.begin() + bond_indexA);
            modesOutBwobond.erase(modesOutBwobond.begin() + bond_indexB);
            extentsOutBwobond.erase(extentsOutBwobond.begin() + bond_indexB);

            std::vector<int32_t> modesOutAB = modesOutAwobond;
            modesOutAB.insert(modesOutAB.end(), modesOutBwobond.begin(), modesOutBwobond.end());

            std::vector<int64_t> extentsOutAB = extentsOutAwobond;
            extentsOutAB.insert(extentsOutAB.end(), extentsOutBwobond.begin(), extentsOutBwobond.end());

            size_t dimOutAB = static_cast<size_t>(dimAwobond) * static_cast<size_t>(dimBwobond);

            void* tensorOutAB;
            HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutAB, dimOutAB * sizeof(complexType)));

            CuTensorNetMethods::ContractTensors(handle_[0],
                                                streams_[0],
                                                modesInAB,
                                                extentsInAB,
                                                qualifiersInAB,
                                                tensorsInAB,
                                                modesOutAB,
                                                dimOutAB,
                                                tensorOutAB,
                                                0.8 / static_cast<double>(numStreams_),
                                                workSpaceLimit_,
                                                {{0, 1}},
                                                workSpacePreference_);

            void* tensorOutA;
            void* tensorOutB;

            HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutA, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
            HANDLE_CUDA_ERROR(cudaMemset(tensorOutA, 0, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

            HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutB, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
            HANDLE_CUDA_ERROR(cudaMemset(tensorOutB, 0, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

            int64_t newABbondExtent = 0;

            CuTensorNetMethods::ApplyTensorSVD(handle_[0],
                                               streams_[0],
                                               svdConfig_[0],
                                               modesOutAB,
                                               extentsOutAB,
                                               modesOutA,
                                               extentsOutA,
                                               modesOutB,
                                               extentsOutB,
                                               {bond_indexA, bond_indexB},
                                               tensorOutAB,
                                               tensorOutA,
                                               tensorOutB,
                                               newABbondExtent,
                                               workSpacePreference_,
                                               verbose);

            siteA_node.virtualExtents_[indexA] = newABbondExtent;
            siteB_node.virtualExtents_[indexB] = newABbondExtent;

            HANDLE_CUDA_ERROR(cudaFree(tensorOutAB));

            HANDLE_CUDA_ERROR(cudaFree(local_tensors[siteA]));
            HANDLE_CUDA_ERROR(cudaFree(local_tensors[siteB]));

            local_tensors[siteA] = tensorOutA;
            local_tensors[siteB] = tensorOutB;
        }

        for(size_t i = 0; i < numSites_; ++i)
        {
            HANDLE_CUDA_ERROR(cudaFree(tensors_[i]));

            tensors_[i] = local_tensors[i];
        }
    }

    TensorNetwork& TensorNetwork::operator+=(const TensorNetwork& Psi)
    {
        AddTensorNet(&Psi);

        return *this;
    }
    
    TensorNetwork TensorNetwork::operator+(const TensorNetwork& Psi) const
    {
        TensorNetwork result(*this);

        result.AddTensorNet(&Psi);

        return result;
    }

    TensorNetwork* TensorNetwork::EvaluateTensorNetProduct(const TensorNetwork* Omega, 
                                                           const std::pair<bool, bool>& transpose,
                                                           const std::pair<bool, bool>& conjugate,
                                                           int64_t maxVirtualExtent, 
                                                           bool verbose) const
    {
        if(check_)
        { 
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                         "The state of the current TN is not initialized.");
            }

            if(Omega->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                         "The state of Omega TN is not initialized.");
            }

            if(numSites_ != Omega->numSites_)
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                         "The number of sites of the current TN differs from that of Omega TN.");
            }

            if(graph_ != Omega->graph_)
            {
                throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                         "The graphs of the current TN and the Omega TN being applied differ from each other.");
            }

            for(size_t i = 0; i < numSites_; ++i)
            {
                const auto& node_i = this->nodes_[i];
                const auto& Omega_node_i = Omega->nodes_[i];

                if(!node_i.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(!Omega_node_i.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of Omega TN.");
                }

                if(node_i.physModes_.size() != Omega_node_i.physModes_.size())
                {
                    throw std::runtime_error("TensorNetwork::ApplyTensorNetProduct: "
                                             "The phys modes of the current TN and Omega TN must have the same phys mode number at site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        TensorNetwork* prodTN = new TensorNetwork(*this);

        int32_t currentMode = prodTN->nextMode_;
        
        prodTN->nextMode_ += Omega->nextMode_;
        
        if(maxVirtualExtent < 1L)
        {
            prodTN->maxVirtualExtent_ += Omega->maxVirtualExtent_;
        }
        else
        {            
            prodTN->maxVirtualExtent_ = maxVirtualExtent;
        }

        auto applying_func = [prodTN, Omega, currentMode, &transpose, &conjugate](size_t i) 
        {
            auto& prodTN_node_i = prodTN->nodes_[i];
            const auto& Omega_node_i = Omega->nodes_[i];
                        
            std::vector<int32_t> Omega_physModes_i = Omega_node_i.physModes_;
            std::transform(Omega_physModes_i.begin(), Omega_physModes_i.end(), Omega_physModes_i.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});

            std::vector<int32_t> OmegaModes = Omega_physModes_i;
            std::vector<int64_t> OmegaExtents = Omega_node_i.physExtents_;
            
            std::vector<int32_t> Omega_virtualModes_i = Omega_node_i.virtualModes_;
            std::vector<int64_t> Omega_virtualExtents_i = Omega_node_i.virtualExtents_;

            std::transform(Omega_virtualModes_i.begin(), Omega_virtualModes_i.end(), Omega_virtualModes_i.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
        
            OmegaModes.insert(OmegaModes.end(), Omega_virtualModes_i.begin(), Omega_virtualModes_i.end());
            OmegaExtents.insert(OmegaExtents.end(), Omega_virtualExtents_i.begin(), Omega_virtualExtents_i.end());

            std::vector<std::vector<int32_t>> modesIn(2);
            std::vector<std::vector<int64_t>> extentsIn(2);
                
            modesIn[0] = prodTN->GetTensorModes(i, false);
            extentsIn[0] = prodTN->GetTensorExtents(i, false);

            modesIn[1] = OmegaModes;
            extentsIn[1] = OmegaExtents;

            std::vector<const void*> tensorsIn(2);
                
            tensorsIn[0] = prodTN->tensors_[i];
            tensorsIn[1] = Omega->tensors_[i];

            std::vector<cutensornetTensorQualifiers_t> qualifiersIn(2);

            qualifiersIn[0].isConjugate = conjugate.first ? 1 : 0;
            qualifiersIn[0].isConstant = 1;
            qualifiersIn[0].requiresGradient = 0;

            qualifiersIn[1].isConjugate = conjugate.second ? 1 : 0;
            qualifiersIn[1].isConstant = 1;
            qualifiersIn[1].requiresGradient = 0;

            std::vector<int32_t> physModesOutSite;
            std::vector<int64_t> physExtentsOutSite;

            size_t numPhysModes = prodTN_node_i.physModes_.size();

            for(size_t j = 0UL; j < numPhysModes; ++j)
            {               
                if(transpose.first)
                {
                    physModesOutSite.push_back(prodTN_node_i.physModes_[(j + numPhysModes / 2UL) % numPhysModes]);
                    physExtentsOutSite.push_back(prodTN_node_i.physExtents_[(j + numPhysModes / 2UL) % numPhysModes]);
                }
                else
                {
                    physModesOutSite.push_back(prodTN_node_i.physModes_[j]);
                    physExtentsOutSite.push_back(prodTN_node_i.physExtents_[j]);
                }

                if(transpose.second)
                {
                    physModesOutSite.push_back(Omega_physModes_i[(j + numPhysModes / 2UL) % numPhysModes]);
                    physExtentsOutSite.push_back(Omega_node_i.physExtents_[(j + numPhysModes / 2UL) % numPhysModes]);
                }
                else
                {
                    physModesOutSite.push_back(Omega_physModes_i[j]);
                    physExtentsOutSite.push_back(Omega_node_i.physExtents_[j]);
                }
            }

            prodTN_node_i.physModes_ = physModesOutSite;
            prodTN_node_i.physExtents_ = physExtentsOutSite;
            prodTN_node_i.extra_virtualModes_ = Omega_virtualModes_i; 
            prodTN_node_i.extra_virtualExtents_ = Omega_virtualExtents_i;

            std::vector<int32_t> modesOutSite = prodTN->GetTensorModes(i);
            std::vector<int64_t> extentsOutSite = prodTN->GetTensorExtents(i);

            size_t dimOut = prodTN->GetTensorSize(i);

            void* tensorOut;
            HANDLE_CUDA_ERROR(cudaMalloc(&tensorOut, dimOut * sizeof(complexType)));

            CuTensorNetMethods::ContractTensors(prodTN->handle_[0],
                                                prodTN->streams_[0],
                                                modesIn,
                                                extentsIn,
                                                qualifiersIn,
                                                tensorsIn,
                                                modesOutSite,
                                                dimOut,
                                                tensorOut,
                                                0.8 / static_cast<double>(prodTN->numStreams_),
                                                prodTN->workSpaceLimit_,
                                                {{0, 1}},
                                                prodTN->workSpacePreference_);

            HANDLE_CUDA_ERROR(cudaFree(prodTN->tensors_[i]));
            prodTN->tensors_[i] = tensorOut;
        };        

        std::vector<bool> appied_nodes(numSites_, false);

        for(const auto& [siteA, siteB] : prodTN->graphTraversalToRoot_)
        {
            if(!appied_nodes.at(siteA))
            {
                appied_nodes[siteA] = true;

                applying_func(siteA);
            }

            if(!appied_nodes.at(siteB))
            {
                appied_nodes[siteB] = true;

                applying_func(siteB);
            }

            try
            {
                prodTN->ExcludeExtraBond(siteA, siteB, 0UL, verbose);
            }
            catch(...){}
        }

        return prodTN;
    }

    complexType TensorNetwork::ComputeMatrixElement(const void* operatorData,
                                                    std::vector<int32_t> operatorModes,
                                                    std::vector<int64_t> operatorExtents,
                                                    const TensorNetwork* ConjPsi,
                                                    size_t thread_num,
                                                    const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes)
    {        
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                         "The state of the current TN is not initialized.");
            }

            std::unordered_map<int32_t, size_t> freq;
            for(const auto& opm : operatorModes) 
            {
                ++freq[opm];
            }

            for(const auto& [key, val] : freq)
            {
                if(val != 2UL) 
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "The operator has modes that are not exactly repeated twice.");
                }
            }

            if(ConjPsi != nullptr) 
            {
                if(ConjPsi->tensors_.empty()) 
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "The state of ConjPsi TN is not initialized.");
                }

                if(numSites_ != ConjPsi->numSites_)
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "The number of sites of the current TN differs from that of ConjPsi TN.");
                }
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                const auto& siteI_node = nodes_[i];

                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(ConjPsi != nullptr)
                {
                    const auto& ConjPsi_node = ConjPsi->nodes_[i];
                    
                    if(!ConjPsi_node.extra_virtualModes_.empty())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                                 "Duplicate bonds should be excluded for site " 
                                                 + std::to_string(i) + " of ConjPsi TN.");
                    }
                
                    if(siteI_node.physModes_.size() != ConjPsi_node.physModes_.size())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                                 "The numbers of phys modes of the current TN and ConjPsi TN are different at site " 
                                                 + std::to_string(i) + ".");
                    }
                }
            }
        }

        int32_t currentMode = nextMode_;

        const size_t numInputTensors = 2UL * numSites_ + 1UL;

        std::vector<std::vector<int32_t>> modesIn(numInputTensors);
        std::vector<std::vector<int64_t>> extentsIn(numInputTensors);

        std::vector<std::vector<int32_t>> ConjPsiphysModes(numSites_);

        size_t operatorNumModes = operatorModes.size();

        size_t opSites = 0UL;

        for(size_t i = 0UL; i < numSites_; ++i) 
        {           
            modesIn[i] = GetTensorModes(i, false);
            extentsIn[i] = GetTensorExtents(i, false);
            
            std::vector<int32_t> conjpsi_physModes_i = nodes_[i].physModes_;

            for(size_t j = 0UL; j < conjpsi_physModes_i.size(); ++j) 
            { 
                auto& mode = conjpsi_physModes_i[j];
                
                size_t enp = 0UL;
                for(size_t k = 0UL; k < operatorNumModes; ++k)
                {
                    auto& opm = operatorModes[k];
                    
                    if(opm == mode)
                    {
                        ++enp;
                        
                        if(check_)
                        {
                            if(nodes_[i].physExtents_[j] != operatorExtents[k])
                            {
                                throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                                         "The operator extent and the corresponding extent of site " 
                                                         + std::to_string(i) + " do not match.");
                            }

                            ++opSites;
                        }
                        
                        if(enp == 2UL)
                        {
                            mode = currentMode++;
                            opm = mode;

                            break;
                        }
                    }
                }
            }

            ConjPsiphysModes[i] = conjpsi_physModes_i;
        }

        if(check_)
        {
            if(opSites != operatorNumModes)
            {
                throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                         "The operator has free modes.");
            }
        }

        for(size_t i = 0UL; i < numSites_; ++i) 
        {                             
            std::vector<int32_t> conjpsi_modes_i = ConjPsiphysModes[i];

            std::vector<int32_t> conjpsi_virtualModes_i = (ConjPsi == nullptr) ? nodes_[i].virtualModes_ : ConjPsi->nodes_[i].virtualModes_;
            std::transform(conjpsi_virtualModes_i.begin(), conjpsi_virtualModes_i.end(), conjpsi_virtualModes_i.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
            conjpsi_modes_i.insert(conjpsi_modes_i.end(), conjpsi_virtualModes_i.begin(), conjpsi_virtualModes_i.end());

            size_t idx = numSites_ + i;

            modesIn[idx] = conjpsi_modes_i;

            if(ConjPsi == nullptr) 
            {
                extentsIn[idx] = extentsIn[i];
            }
            else 
            {                
                extentsIn[idx] = ConjPsi->GetTensorExtents(i, false);
            }            
        }

        ConjPsiphysModes.clear();    

        size_t opIdx = 2UL * numSites_;

        modesIn[opIdx] = operatorModes;
        extentsIn[opIdx] = operatorExtents;

        std::vector<const void*> tensorsIn(numInputTensors);
        
        for(size_t i = 0UL; i < numSites_; ++i) 
        {
            tensorsIn[i] = tensors_[i];

            if(ConjPsi == nullptr)
            {
                tensorsIn[numSites_ + i] = tensors_[i];
            }
            else
            {
                tensorsIn[numSites_ + i] = ConjPsi->tensors_[i];
            }
        }

        tensorsIn[opIdx] = operatorData;

        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputTensors);

        for(size_t i = 0UL; i < numSites_; ++i) 
        {
            qualifiersIn[i].isConjugate = 0;
            qualifiersIn[i].isConstant = 1;
            qualifiersIn[i].requiresGradient = 0;

            size_t idx = numSites_ + i;
            qualifiersIn[idx].isConjugate = 1;
            qualifiersIn[idx].isConstant = 1;
            qualifiersIn[idx].requiresGradient = 0;
        }

        qualifiersIn[opIdx].isConjugate = 0;
        qualifiersIn[opIdx].isConstant = 1;
        qualifiersIn[opIdx].requiresGradient = 0;

        void* result_device;
        HANDLE_CUDA_ERROR(cudaMalloc(&result_device, sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesIn,
                                            extentsIn,
                                            qualifiersIn,
                                            tensorsIn,
                                            {},
                                            1UL,
                                            result_device,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {},
                                            workSpacePreference_,
                                            optimizerAttributes,
                                            CuTensorNetMethods::MPI_);

        complexType result_host;
        HANDLE_CUDA_ERROR(cudaMemcpy((void*)(&result_host), result_device, sizeof(complexType), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaFree(result_device));

        return result_host;
    }

    complexType TensorNetwork::ComputeMatrixElement(const TensorNetwork* Omega,
                                                    const TensorNetwork* ConjPsi,
                                                    size_t thread_num,
                                                    const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                         "The state of the current TN is not initialized.");
            }

            if(ConjPsi != nullptr) 
            {
                if(ConjPsi->tensors_.empty()) 
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "The state of ConjPsi TN is not initialized.");
                }

                if(numSites_ != ConjPsi->numSites_)
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "The number of sites of the current TN differs from that of ConjPsi TN.");
                }
            }

            if(Omega->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                         "The state of Omega TN is not initialized.");
            }

            if(numSites_ != Omega->numSites_)
            {
                throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                         "The number of sites of the current TN differs from that of Omega TN.");
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                const auto& siteI_node = nodes_[i];
                const auto& Omega_node = Omega->nodes_[i];

                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(ConjPsi != nullptr)
                {
                    if(!ConjPsi->nodes_[i].extra_virtualModes_.empty())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                                 "Duplicate bonds should be excluded for site " 
                                                 + std::to_string(i) + " of ConjPsi TN.");
                    }
                }

                if(!Omega_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of Omega TN.");
                }

                if(ConjPsi != nullptr)
                {
                    if(siteI_node.physModes_.size() + ConjPsi->nodes_[i].physModes_.size() != Omega_node.physModes_.size())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                                 "The total number of phys modes of the current TN and ConjPsi TN must be "
                                                 "equal to that of Omega TN at site " + std::to_string(i) + ".");
                    }
                }
                else
                {
                    if(siteI_node.physModes_.size() * 2UL != Omega_node.physModes_.size())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeMatrixElement: "
                                                 "The double number of phys modes of the current TN must be "
                                                 "equal to that of Omega TN at site " + std::to_string(i) + ".");
                    }
                }
            }
        }

        int32_t currentMode = nextMode_;
        int32_t currentConjPsiMode = (ConjPsi == nullptr) ? nextMode_ : ConjPsi->nextMode_;

        const size_t numInputTensors = 3UL * numSites_;

        std::vector<std::vector<int32_t>> modesIn(numInputTensors);
        std::vector<std::vector<int64_t>> extentsIn(numInputTensors);

        std::vector<std::vector<int32_t>> ConjPsiphysModes(numSites_);
        
        for(size_t i = 0; i < numSites_; ++i)
        {
            size_t physModeSize;

            if(ConjPsi != nullptr) 
            {
                physModeSize = ConjPsi->nodes_[i].physModes_.size();
            }
            else
            {
                physModeSize = nodes_[i].physModes_.size();
            }

            ConjPsiphysModes[i].resize(physModeSize);

            for(auto& el : ConjPsiphysModes[i])
            {
                el = currentMode++;
            }
        }
        
        for(size_t i = 0; i < numSites_; ++i) 
        {                   
            const auto& siteI_node = nodes_[i];

            modesIn[i] = GetTensorModes(i, false);
            extentsIn[i] = GetTensorExtents(i, false);

            std::vector<int32_t> conjpsi_modes_i = ConjPsiphysModes[i];

            std::vector<int32_t> conjpsi_virtualModes_i = (ConjPsi == nullptr) ? siteI_node.virtualModes_ : ConjPsi->nodes_[i].virtualModes_;
            std::transform(conjpsi_virtualModes_i.begin(), conjpsi_virtualModes_i.end(), conjpsi_virtualModes_i.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
            conjpsi_modes_i.insert(conjpsi_modes_i.end(), conjpsi_virtualModes_i.begin(), conjpsi_virtualModes_i.end());

            size_t idx1 = numSites_ + i;
            
            modesIn[idx1] = conjpsi_modes_i;

            if(ConjPsi == nullptr) 
            {
                extentsIn[idx1] = extentsIn[i];
            }
            else 
            {                
                extentsIn[idx1] = ConjPsi->GetTensorExtents(i, false);
            }

            std::vector<int32_t> operator_modes_i = siteI_node.physModes_;
            operator_modes_i.insert(operator_modes_i.end(), ConjPsiphysModes[i].begin(), ConjPsiphysModes[i].end());

            std::vector<int32_t> operator_virtualModes_i = Omega->nodes_[i].virtualModes_;
            std::transform(operator_virtualModes_i.begin(), operator_virtualModes_i.end(), operator_virtualModes_i.begin(), 
                          [&currentMode, &currentConjPsiMode](int32_t x) -> int32_t {return x + currentMode + currentConjPsiMode;});
            operator_modes_i.insert(operator_modes_i.end(), operator_virtualModes_i.begin(), operator_virtualModes_i.end());

            size_t idx2 = 2 * numSites_ + i;

            modesIn[idx2] = operator_modes_i;
            extentsIn[idx2] = Omega->GetTensorExtents(i, false);
        }
        
        std::vector<const void*> tensorsIn(numInputTensors);
        
        for(size_t i = 0; i < numSites_; ++i) 
        {
            tensorsIn[i] = tensors_[i];

            if(ConjPsi == nullptr)
            {
                tensorsIn[numSites_ + i] = tensors_[i];
            }
            else
            {
                tensorsIn[numSites_ + i] = ConjPsi->tensors_[i];
            }

            tensorsIn[2 * numSites_ + i] = Omega->tensors_[i];
        }

        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputTensors);

        for(size_t i = 0; i < numSites_; ++i) 
        {
            qualifiersIn[i].isConjugate = 0;
            qualifiersIn[i].isConstant = 1;
            qualifiersIn[i].requiresGradient = 0;

            size_t idx1 = numSites_ + i;
            qualifiersIn[idx1].isConjugate = 1;
            qualifiersIn[idx1].isConstant = 1;
            qualifiersIn[idx1].requiresGradient = 0;

            size_t idx2 = 2 * numSites_ + i;
            qualifiersIn[idx2].isConjugate = 0;
            qualifiersIn[idx2].isConstant = 1;
            qualifiersIn[idx2].requiresGradient = 0;
        }

        void* result_device;
        HANDLE_CUDA_ERROR(cudaMalloc(&result_device, sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesIn,
                                            extentsIn,
                                            qualifiersIn,
                                            tensorsIn,
                                            {},
                                            1UL,
                                            result_device,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {},
                                            workSpacePreference_,
                                            optimizerAttributes,
                                            CuTensorNetMethods::MPI_);

        complexType result_host;
        HANDLE_CUDA_ERROR(cudaMemcpy((void*)(&result_host), result_device, sizeof(complexType), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaFree(result_device));

        return result_host;
    }

    complexType TensorNetwork::ComputeOperatorTrace(const void* operatorData,
                                                    std::vector<int32_t> operatorModes,
                                                    std::vector<int64_t> operatorExtents,
                                                    size_t thread_num,
                                                    const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                         "The state of the current TN is not initialized.");
            }

            std::unordered_map<int32_t, size_t> freq;
            for(const auto& opm : operatorModes) 
            {
                ++freq[opm];
            }

            for(const auto& [key, val] : freq)
            {
                if(val != 2UL) 
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "The operator has modes that are not exactly repeated twice.");
                }
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                const auto& siteI_node = nodes_[i];

                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(siteI_node.physModes_.size() % 2UL != 0)
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "The current TN must be a operator.");
                }
            }
        }

        const size_t numInputTensors = numSites_ + 1UL;

        std::vector<std::vector<int32_t>> modesIn(numInputTensors);
        std::vector<std::vector<int64_t>> extentsIn(numInputTensors);

        size_t operatorNumModes = operatorModes.size();

        size_t opSites = 0UL;

        for(size_t i = 0UL; i < numSites_; ++i) 
        {           
            std::vector<int32_t> rho_modes_i = GetTensorModes(i, false);
            std::vector<int64_t> rho_extents_i = GetTensorExtents(i, false);

            size_t num_physModes = nodes_[i].physModes_.size();

            std::vector<bool> used_physModes(num_physModes, false);

            for(size_t j = 0UL; j < num_physModes; ++j) 
            {           
                const auto& mode = rho_modes_i[j];

                size_t enp = 0UL;
                for(size_t k = 0UL; k < operatorNumModes; ++k)
                {
                    auto& opm = operatorModes[k];

                    if(opm == mode)
                    {
                        ++enp;

                        if(check_)
                        {
                            if(rho_extents_i[j] != operatorExtents[k])
                            {
                                throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                                         "The operator extent and the corresponding extent of site " 
                                                         + std::to_string(i) + " do not match.");
                            }

                            ++opSites;
                        }
                        
                        if(enp == 2UL)
                        {
                            size_t neighbor = (j < num_physModes / 2UL) ? j + num_physModes / 2UL : j - num_physModes / 2UL;

                            opm = rho_modes_i[neighbor];

                            used_physModes[j] = true;
                            used_physModes[neighbor] = true;

                            break;
                        }
                    }
                }
            }

            size_t half_num_physModes = num_physModes / 2UL;

            for(size_t j = 0; j < half_num_physModes; ++j) 
            {                               
                if(!used_physModes[j])
                {
                    rho_modes_i[j + half_num_physModes] = rho_modes_i[j];
                }
            }

            modesIn[i] = rho_modes_i;
            extentsIn[i] = rho_extents_i;
        }

        if(check_)
        {
            if(opSites != operatorNumModes)
            {
                throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                         "The operator has free modes.");
            }
        } 

        modesIn[numSites_] = operatorModes;
        extentsIn[numSites_] = operatorExtents;
        
        std::vector<const void*> tensorsIn(numInputTensors);
        
        for(size_t i = 0; i < numSites_; ++i) 
        {
            tensorsIn[i] = tensors_[i];
        }

        tensorsIn[numSites_] = operatorData;

        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputTensors);

        for(size_t i = 0UL; i <= numSites_; ++i) 
        {
            qualifiersIn[i].isConjugate = 0;
            qualifiersIn[i].isConstant = 1;
            qualifiersIn[i].requiresGradient = 0;
        }

        void* result_device;
        HANDLE_CUDA_ERROR(cudaMalloc(&result_device, sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesIn,
                                            extentsIn,
                                            qualifiersIn,
                                            tensorsIn,
                                            {},
                                            1UL,
                                            result_device,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {},
                                            workSpacePreference_,
                                            optimizerAttributes,
                                            CuTensorNetMethods::MPI_);

        complexType result_host;
        HANDLE_CUDA_ERROR(cudaMemcpy((void*)(&result_host), result_device, sizeof(complexType), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaFree(result_device));

        return result_host;
    }

    complexType TensorNetwork::ComputeOperatorTrace(const TensorNetwork* Omega,
                                                    size_t thread_num,
                                                    const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                         "The state of the current TN is not initialized.");
            }

            if(Omega != nullptr)
            {
                if(Omega->tensors_.empty()) 
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "The state of Omega TN is not initialized.");
                }

                if(numSites_ != Omega->numSites_)
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "The number of sites of the current TN differs from that of Omega TN.");
                }
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                const auto& siteI_node = nodes_[i];

                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }
                if(siteI_node.physModes_.size() % 2UL != 0)
                {
                    throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                             "The current TN must be a operator.");
                }

                if(Omega != nullptr)
                {
                    const auto& Omega_node = Omega->nodes_[i];
                    
                    if(!Omega_node.extra_virtualModes_.empty())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                                 "Duplicate bonds should be excluded for site " 
                                                 + std::to_string(i) + " of Omega TN.");
                    }

                    if(Omega_node.physModes_.size() % 2UL != 0)
                    {
                        throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                                 "The Omega TN must be a operator.");
                    }

                    if(siteI_node.physModes_.size() != Omega_node.physModes_.size())
                    {
                        throw std::runtime_error("TensorNetwork::ComputeOperatorTrace: "
                                                 "The phys modes of the current TN and Omega TN are not suitable at site " 
                                                 + std::to_string(i) + ".");
                    }
                }
            }
        }

        int32_t currentMode = nextMode_;

        const size_t numInputTensors = numSites_ * ((Omega != nullptr) ? 2UL : 1UL);

        std::vector<std::vector<int32_t>> modesIn(numInputTensors);
        std::vector<std::vector<int64_t>> extentsIn(numInputTensors);

        if(Omega == nullptr)
        {
            for(size_t i = 0; i < numSites_; ++i) 
            {                   
                std::vector<int32_t> rho_modes_i = GetTensorModes(i, false);

                size_t half_num_physModes = nodes_[i].physModes_.size() / 2UL;

                for(size_t j = 0; j < half_num_physModes; ++j) 
                {                               
                    rho_modes_i[j + half_num_physModes] = rho_modes_i[j];
                }

                modesIn[i] = rho_modes_i;
                extentsIn[i] = GetTensorExtents(i, false);
            }
        }
        else
        {
            for(size_t i = 0; i < numSites_; ++i) 
            {
                modesIn[i] = GetTensorModes(i, false);
                extentsIn[i] = GetTensorExtents(i, false);

                std::vector<int32_t> operator_modes_i = nodes_[i].physModes_;
                std::reverse(operator_modes_i.begin(), operator_modes_i.end());

                std::vector<int32_t> operator_virtualModes_i = Omega->nodes_[i].virtualModes_;
                std::transform(operator_virtualModes_i.begin(), operator_virtualModes_i.end(), operator_virtualModes_i.begin(), 
                              [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
                operator_modes_i.insert(operator_modes_i.end(), operator_virtualModes_i.begin(), operator_virtualModes_i.end());

                size_t idx2 = numSites_ + i;

                modesIn[idx2] = operator_modes_i;
                extentsIn[idx2] = Omega->GetTensorExtents(i, false);
            }
        }
        
        std::vector<const void*> tensorsIn(numInputTensors);
        
        for(size_t i = 0; i < numSites_; ++i) 
        {
            tensorsIn[i] = tensors_[i];

            if(Omega != nullptr)
            {   
                tensorsIn[numSites_ + i] = Omega->tensors_[i];
            }
        }

        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputTensors);

        for(size_t i = 0; i < numSites_; ++i) 
        {
            qualifiersIn[i].isConjugate = 0;
            qualifiersIn[i].isConstant = 1;
            qualifiersIn[i].requiresGradient = 0;

            if(Omega != nullptr)
            {
                size_t idx2 = numSites_ + i;
                
                qualifiersIn[idx2].isConjugate = 0;
                qualifiersIn[idx2].isConstant = 1;
                qualifiersIn[idx2].requiresGradient = 0;
            }
        }

        void* result_device;
        HANDLE_CUDA_ERROR(cudaMalloc(&result_device, sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesIn,
                                            extentsIn,
                                            qualifiersIn,
                                            tensorsIn,
                                            {},
                                            1UL,
                                            result_device,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {},
                                            workSpacePreference_,
                                            optimizerAttributes,
                                            CuTensorNetMethods::MPI_);

        complexType result_host;
        HANDLE_CUDA_ERROR(cudaMemcpy((void*)(&result_host), result_device, sizeof(complexType), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaFree(result_device));

        return result_host;
    }

    void TensorNetwork::ExcludeExtraBond(size_t siteA, 
                                         size_t siteB,
                                         size_t thread_num,
                                         bool verbose)
    {        
        if(check_)
        {
            if((siteB >= numSites_) || (siteA >= numSites_))
            {
                throw std::invalid_argument("TensorNetwork::ExcludeExtraBond: "
                                            "Site index can not exceed maximal number of sites.");
            }

            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::ExcludeExtraBond: "
                                         "The state is not initialized.");
            }
        }

        auto& siteA_node = nodes_[siteA];
        auto& siteB_node = nodes_[siteB];

        auto itA = siteA_node.neighbors_.find(siteB);
        auto itB = siteB_node.neighbors_.find(siteA);

        if(check_)
        {
            if((itA == siteA_node.neighbors_.end()) || (itB == siteB_node.neighbors_.end())) 
            {
                throw std::invalid_argument("TensorNetwork::ExcludeExtraBond: "
                                            "Site " + std::to_string(siteA) + 
                                            " must be a neighbor of site " 
                                            + std::to_string(siteB) + ".");
            }
            
            if(siteA_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::ExcludeExtraBond: "
                                         "There are not any duplicate bonds for site " 
                                         + std::to_string(siteA) + ".");
            }

            if(siteB_node.extra_virtualModes_.empty())
            {
                throw std::runtime_error("TensorNetwork::ExcludeExtraBond: "
                                         "There are not any duplicate bonds for site " 
                                         + std::to_string(siteB) + ".");
            }
        }

        size_t indexA = itA->second;
        size_t indexB = itB->second;

        const auto& exmodesA = siteA_node.extra_virtualModes_;
        const auto& exmodesB = siteB_node.extra_virtualModes_;

        std::unordered_set<int32_t> exModesToSetB(exmodesB.begin(), exmodesB.end());
        std::vector<size_t> indicesToRemoveA;

        for(size_t i = 0; i < exmodesA.size(); ++i) 
        {
            if (exModesToSetB.count(exmodesA[i])) 
            {
                indicesToRemoveA.push_back(i);
            }
        }

        std::unordered_set<int32_t> exModesToSetA(exmodesA.begin(), exmodesA.end());
        std::vector<size_t> indicesToRemoveB;
        
        for(size_t i = 0; i < exmodesB.size(); ++i) 
        {
            if (exModesToSetA.count(exmodesB[i])) 
            {
                indicesToRemoveB.push_back(i);
            }
        }

        auto shouldSkip = [](size_t idx, const std::vector<size_t>& toRemove) 
        {
            return std::binary_search(toRemove.begin(), toRemove.end(), idx);
        };

        std::vector<int32_t> extra_virtualModesOutA;
        std::vector<int64_t> extra_virtualExtentsOutA;

        for(size_t i = 0; i < exmodesA.size(); ++i) 
        {
            if(!shouldSkip(i, indicesToRemoveA)) 
            {
                extra_virtualModesOutA.push_back(exmodesA[i]);
                extra_virtualExtentsOutA.push_back(siteA_node.extra_virtualExtents_[i]);
            }
        }

        std::vector<int32_t> extra_virtualModesOutB;
        std::vector<int64_t> extra_virtualExtentsOutB;

        for(size_t i = 0; i < exmodesB.size(); ++i) 
        {
            if(!shouldSkip(i, indicesToRemoveB)) 
            {
                extra_virtualModesOutB.push_back(exmodesB[i]);
                extra_virtualExtentsOutB.push_back(siteB_node.extra_virtualExtents_[i]);
            }
        }

        if(check_)
        {
            if((extra_virtualModesOutA.size() == exmodesA.size()) || (extra_virtualModesOutB.size() == exmodesB.size()))
            {
                throw std::runtime_error("TensorNetwork::ExcludeExtraBond: "
                                         "There are not any duplicate bonds between sites (" 
                                         + std::to_string(siteA) + ", " + std::to_string(siteB) + ").");
            }
        }

        std::vector<std::vector<int32_t>> modesInAB(2);
        std::vector<std::vector<int64_t>> extentsInAB(2);
        
        modesInAB[0] = GetTensorModes(siteA);
        extentsInAB[0] = GetTensorExtents(siteA);

        modesInAB[1] = GetTensorModes(siteB);
        extentsInAB[1] = GetTensorExtents(siteB);

        std::vector<const void*> tensorsInAB(2);
        
        tensorsInAB[0] = tensors_[siteA];
        tensorsInAB[1] = tensors_[siteB];

        std::vector<cutensornetTensorQualifiers_t> qualifiersInAB(2);

        qualifiersInAB[0].isConjugate = 0;
        qualifiersInAB[0].isConstant = 1;
        qualifiersInAB[0].requiresGradient = 0;

        qualifiersInAB[1].isConjugate = 0;
        qualifiersInAB[1].isConstant = 1;
        qualifiersInAB[1].requiresGradient = 0;

        siteA_node.extra_virtualModes_ = extra_virtualModesOutA;
        siteA_node.extra_virtualExtents_ = extra_virtualExtentsOutA;

        siteB_node.extra_virtualModes_ = extra_virtualModesOutB;
        siteB_node.extra_virtualExtents_ = extra_virtualExtentsOutB;

        int64_t dimAwobond = static_cast<int64_t>(GetTensorSize(siteA)) / siteA_node.virtualExtents_[indexA];
        int64_t dimBwobond = static_cast<int64_t>(GetTensorSize(siteB)) / siteB_node.virtualExtents_[indexB];

        int64_t extentABbond = std::min({dimAwobond, dimBwobond, maxVirtualExtent_});

        siteA_node.virtualExtents_[indexA] = extentABbond;
        siteB_node.virtualExtents_[indexB] = extentABbond;

        std::vector<int32_t> modesOutA = GetTensorModes(siteA);
        std::vector<int64_t> extentsOutA = GetTensorExtents(siteA);

        std::vector<int32_t> modesOutB = GetTensorModes(siteB);
        std::vector<int64_t> extentsOutB = GetTensorExtents(siteB);

        std::vector<int32_t> modesOutAwobond = modesOutA;
        std::vector<int32_t> modesOutBwobond = modesOutB;
        std::vector<int64_t> extentsOutAwobond = extentsOutA;
        std::vector<int64_t> extentsOutBwobond = extentsOutB;

        size_t bond_indexA = siteA_node.physModes_.size() + indexA;
        size_t bond_indexB = siteB_node.physModes_.size() + indexB;

        modesOutAwobond.erase(modesOutAwobond.begin() + bond_indexA);
        extentsOutAwobond.erase(extentsOutAwobond.begin() + bond_indexA);
        modesOutBwobond.erase(modesOutBwobond.begin() + bond_indexB);
        extentsOutBwobond.erase(extentsOutBwobond.begin() + bond_indexB);

        std::vector<int32_t> modesOutAB = modesOutAwobond;
        modesOutAB.insert(modesOutAB.end(), modesOutBwobond.begin(), modesOutBwobond.end());

        std::vector<int64_t> extentsOutAB = extentsOutAwobond;
        extentsOutAB.insert(extentsOutAB.end(), extentsOutBwobond.begin(), extentsOutBwobond.end());

        size_t dimOutAB = static_cast<size_t>(dimAwobond) * static_cast<size_t>(dimBwobond);

        void* tensorOutAB;
        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutAB, dimOutAB * sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesInAB,
                                            extentsInAB,
                                            qualifiersInAB,
                                            tensorsInAB,
                                            modesOutAB,
                                            dimOutAB,
                                            tensorOutAB,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {{0, 1}},
                                            workSpacePreference_);

        void* tensorOutA;
        void* tensorOutB;

        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutA, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemset(tensorOutA, 0, static_cast<size_t>(dimAwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

        HANDLE_CUDA_ERROR(cudaMalloc(&tensorOutB, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));
        HANDLE_CUDA_ERROR(cudaMemset(tensorOutB, 0, static_cast<size_t>(dimBwobond) * static_cast<size_t>(extentABbond) * sizeof(complexType)));

        int64_t newABbondExtent = 0;

        CuTensorNetMethods::ApplyTensorSVD(handle_.at(thread_num),
                                           streams_.at(thread_num),
                                           svdConfig_.at(thread_num),
                                           modesOutAB,
                                           extentsOutAB,
                                           modesOutA,
                                           extentsOutA,
                                           modesOutB,
                                           extentsOutB,
                                           {bond_indexA, bond_indexB},
                                           tensorOutAB,
                                           tensorOutA,
                                           tensorOutB,
                                           newABbondExtent,
                                           workSpacePreference_,
                                           verbose);
        
        siteA_node.virtualExtents_[indexA] = newABbondExtent;
        siteB_node.virtualExtents_[indexB] = newABbondExtent;

        HANDLE_CUDA_ERROR(cudaFree(tensorOutAB));

        HANDLE_CUDA_ERROR(cudaFree(tensors_[siteA]));
        HANDLE_CUDA_ERROR(cudaFree(tensors_[siteB]));

        tensors_[siteA] = tensorOutA;
        tensors_[siteB] = tensorOutB;
    }

    std::pair<void*, TensorDescriptor> TensorNetwork::GetDensityMatrix(const std::vector<size_t>& keep_nodes,
                                                                       bool circuit_order,
                                                                       size_t thread_num,
                                                                       const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::GetDensityMatrix: "
                                         "The state is not initialized.");
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                if(!nodes_[i].extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::GetDensityMatrix: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        int32_t currentMode = nextMode_;

        const size_t numInputTensors = 2UL * numSites_;

        std::vector<std::vector<int32_t>> modesIn(numInputTensors);
        std::vector<std::vector<int64_t>> extentsIn(numInputTensors);

        std::vector<std::vector<int32_t>> ConjPsiphysModes(numSites_);

        std::vector<int32_t> rhoModes_left;
        std::vector<int64_t> rhoExtents_left;
        std::vector<int32_t> rhoModes_right;
        std::vector<int64_t> rhoExtents_right;
        
        for(size_t i = 0; i < numSites_; ++i) 
        {
            const auto& siteI_node = nodes_[i];

            modesIn[i] = GetTensorModes(i, false);
            extentsIn[i] = GetTensorExtents(i, false);

            std::vector<int32_t> conjPsi_physModes_i = siteI_node.physModes_;

            auto it = std::find(keep_nodes.begin(), keep_nodes.end(), i);

            if(it != keep_nodes.end())
            {
                rhoModes_left.insert(rhoModes_left.end(), siteI_node.physModes_.begin(), siteI_node.physModes_.end());
                rhoExtents_left.insert(rhoExtents_left.end(), siteI_node.physExtents_.begin(), siteI_node.physExtents_.end());

                for(auto& el : conjPsi_physModes_i)
                {
                    el = currentMode++;
                }

                rhoModes_right.insert(rhoModes_right.end(), conjPsi_physModes_i.begin(), conjPsi_physModes_i.end());
                rhoExtents_right.insert(rhoExtents_right.end(), siteI_node.physExtents_.begin(), siteI_node.physExtents_.end());
            }

            ConjPsiphysModes[i] = conjPsi_physModes_i;
        }
        
        for(size_t i = 0; i < numSites_; ++i) 
        {
            std::vector<int32_t> conjPsi_modes_i = ConjPsiphysModes[i];
            std::vector<int32_t> conjPsi_virtualModes_i = nodes_[i].virtualModes_;

            std::transform(conjPsi_virtualModes_i.begin(), conjPsi_virtualModes_i.end(), conjPsi_virtualModes_i.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});

            conjPsi_modes_i.insert(conjPsi_modes_i.end(), conjPsi_virtualModes_i.begin(), conjPsi_virtualModes_i.end());

            size_t idx = numSites_ + i;

            modesIn[idx] = conjPsi_modes_i;
            extentsIn[idx] = extentsIn[i];
        }

        ConjPsiphysModes.clear();

        std::vector<int32_t> rhoModes;
        std::vector<int64_t> rhoExtents;

        if(circuit_order)
        {
            rhoModes.insert(rhoModes.end(), rhoModes_left.begin(), rhoModes_left.end());
            rhoExtents.insert(rhoExtents.end(), rhoExtents_left.begin(), rhoExtents_left.end());

            rhoModes.insert(rhoModes.end(), rhoModes_right.begin(), rhoModes_right.end());
            rhoExtents.insert(rhoExtents.end(), rhoExtents_right.begin(), rhoExtents_right.end());
        }
        else
        {
            rhoModes.insert(rhoModes.end(), rhoModes_right.begin(), rhoModes_right.end());
            rhoExtents.insert(rhoExtents.end(), rhoExtents_right.begin(), rhoExtents_right.end());

            rhoModes.insert(rhoModes.end(), rhoModes_left.begin(), rhoModes_left.end());
            rhoExtents.insert(rhoExtents.end(), rhoExtents_left.begin(), rhoExtents_left.end());
        }

        rhoModes_left.clear();
        rhoExtents_left.clear();
        rhoModes_right.clear();
        rhoExtents_right.clear();

        std::vector<const void*> tensorsIn(numInputTensors);
        
        for(size_t i = 0; i < numSites_; ++i) 
        {
            tensorsIn[i] = tensors_[i];
            tensorsIn[numSites_ + i] = tensors_[i];
        }

        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputTensors);

        for(size_t i = 0; i < numSites_; ++i) 
        {
            qualifiersIn[i].isConjugate = 0;
            qualifiersIn[i].isConstant = 1;
            qualifiersIn[i].requiresGradient = 0;

            size_t idx = numSites_ + i;
            qualifiersIn[idx].isConjugate = 1;
            qualifiersIn[idx].isConstant = 1;
            qualifiersIn[idx].requiresGradient = 0;
        }

        size_t rho_dim = static_cast<size_t>(std::accumulate(rhoExtents.begin(),
                                             rhoExtents.end(),
                                             1L,
                                             [](int64_t acc, const int64_t& current) {return acc * current;}));
        
        void* rho_tensor;
        HANDLE_CUDA_ERROR(cudaMalloc(&rho_tensor, rho_dim * sizeof(complexType)));

        CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                            streams_.at(thread_num),
                                            modesIn,
                                            extentsIn,
                                            qualifiersIn,
                                            tensorsIn,
                                            rhoModes,
                                            rho_dim,
                                            rho_tensor,
                                            0.8 / static_cast<double>(numStreams_),
                                            workSpaceLimit_,
                                            {},
                                            workSpacePreference_,
                                            optimizerAttributes,
                                            CuTensorNetMethods::MPI_);

        return {rho_tensor, TensorDescriptor(rhoModes, rhoExtents)};
    }

    std::pair<std::vector<const void*>, TensorNetDescriptor> TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator(const TensorNetwork* Omega,
                                                                                                                           CachedLeaves& cache,
                                                                                                                           const TensorNetwork* ConjPsi,
                                                                                                                           const std::vector<size_t>& keep_nodes,
                                                                                                                           size_t thread_num,
                                                                                                                           const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes)
    {
        if(check_)
        {
            if(keep_nodes.empty())
            {
                throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                         "keep_nodes cannot be empty.");
            }
            
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                         "The state of the current TN is not initialized.");
            }

            if(Omega->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                         "The state of Omega TN is not initialized.");
            }

            if(numSites_ != Omega->numSites_)
            {
                throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                         "The number of sites of the current TN differs from that of Omega TN.");
            }

            if(graph_ != Omega->graph_)
            {
                throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                         "The graphs of the current TN and the Omega TN differ from each other.");
            }

            if(ConjPsi != nullptr) 
            {
                if(ConjPsi->tensors_.empty()) 
                {
                    throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                             "The state of ConjPsi TN is not initialized.");
                }

                if(numSites_ != ConjPsi->numSites_)
                {
                    throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                             "The number of sites of the current TN differs from that of ConjPsi TN.");
                }

                if(graph_ != ConjPsi->graph_)
                {
                    throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                             "The graphs of the current TN and the ConjPsi TN differ from each other.");
                }
            }

            for(size_t i = 0; i < numSites_; ++i) 
            {
                const auto& siteI_node = nodes_[i];
                const auto& Omega_node = Omega->nodes_[i];

                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(!Omega_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of Omega TN.");
                }

                if(siteI_node.physModes_.size() * 2UL != Omega_node.physModes_.size())
                {
                    throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                             "The double number of phys modes of the current TN must be "
                                             "equal to that of Omega TN at site " + std::to_string(i) + ".");
                }

                if(ConjPsi != nullptr)
                {
                    const auto& ConjPsi_node = ConjPsi->nodes_[i];
                    
                    if(!ConjPsi_node.extra_virtualModes_.empty())
                    {
                        throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                                 "Duplicate bonds should be excluded for site " 
                                                 + std::to_string(i) + " of ConjPsi TN.");
                    }
                
                    if(siteI_node.physModes_.size() != ConjPsi_node.physModes_.size())
                    {
                        throw std::runtime_error("TensorNetwork::EvaluateTensorNetDescriptorOfEffectiveOperator: "
                                                 "The numbers of phys modes of the current TN and ConjPsi TN are different at site " 
                                                 + std::to_string(i) + ".");
                    }
                }
            }
        }

        int32_t currentMode = nextMode_;
        int32_t currentConjPsiMode = (ConjPsi == nullptr) ? nextMode_ : ConjPsi->nextMode_;

        const size_t numkeepNodes = keep_nodes.size();
        const size_t numprocessingNodes = numSites_ - numkeepNodes;

        const size_t numInputTensors = 3UL * numSites_ - 2UL * numkeepNodes + 1UL;

        std::vector<std::vector<int32_t>> modesIn(numInputTensors);
        std::vector<std::vector<int64_t>> extentsIn(numInputTensors);

        std::vector<std::vector<int32_t>> ConjPsiphysModes(numSites_);

        std::vector<size_t> processing_nodes;
        processing_nodes.reserve(numprocessingNodes);

        for(size_t i = 0; i < numSites_; ++i)
        {
            auto it = std::find(keep_nodes.begin(), keep_nodes.end(), i);

            if(it == keep_nodes.end())
            {
                processing_nodes.push_back(i);
            }
        }

        std::vector<const void*> tensorsIn(numInputTensors);
        std::vector<cutensornetTensorQualifiers_t> qualifiersIn(numInputTensors);

        for(size_t j = 0; j < numSites_; ++j) 
        {                   
            const size_t& i = (j < numprocessingNodes) ? processing_nodes[j] : keep_nodes[j - numprocessingNodes];

            if(j < numprocessingNodes)
            {
                modesIn[j] = GetTensorModes(i, false);
                extentsIn[j] = GetTensorExtents(i, false);
                
                tensorsIn[j] = tensors_[i];

                qualifiersIn[j].isConjugate = 0;
                qualifiersIn[j].isConstant = 1;
                qualifiersIn[j].requiresGradient = 0;
            }

            size_t physModeSize;

            if(ConjPsi != nullptr) 
            {
                physModeSize = ConjPsi->nodes_[i].physModes_.size();
            }
            else
            {
                physModeSize = nodes_[i].physModes_.size();
            }

            ConjPsiphysModes[j].resize(physModeSize);

            for(auto& el : ConjPsiphysModes[j])
            {
                el = currentMode++;
            }
        }

        std::vector<size_t> accessPoint(numSites_);

        for(size_t j = 0; j < numSites_; ++j) 
        {                   
            const size_t& i = (j < numprocessingNodes) ? processing_nodes[j] : keep_nodes[j - numprocessingNodes];

            accessPoint[i] = j;

            if(j < numprocessingNodes)
            {
                std::vector<int32_t> conjpsi_modes_j = ConjPsiphysModes[j];

                std::vector<int32_t> conjpsi_virtualModes_j = (ConjPsi == nullptr) ? nodes_[i].virtualModes_ : ConjPsi->nodes_[i].virtualModes_;
                std::transform(conjpsi_virtualModes_j.begin(), conjpsi_virtualModes_j.end(), conjpsi_virtualModes_j.begin(), 
                              [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
                conjpsi_modes_j.insert(conjpsi_modes_j.end(), conjpsi_virtualModes_j.begin(), conjpsi_virtualModes_j.end());               

                size_t idx1 = numprocessingNodes + j;

                modesIn[idx1] = conjpsi_modes_j;

                if(ConjPsi == nullptr) 
                {
                    extentsIn[idx1] = extentsIn[j];
                    tensorsIn[idx1] = tensors_[i];
                }
                else 
                {                
                    extentsIn[idx1] = ConjPsi->GetTensorExtents(i, false);  
                    tensorsIn[idx1] = ConjPsi->tensors_[i];
                }

                qualifiersIn[idx1].isConjugate = 1;
                qualifiersIn[idx1].isConstant = 1;
                qualifiersIn[idx1].requiresGradient = 0;
            }

            std::vector<int32_t> operator_modes_j = nodes_[i].physModes_;
            operator_modes_j.insert(operator_modes_j.end(), ConjPsiphysModes[j].begin(), ConjPsiphysModes[j].end());

            std::vector<int32_t> operator_virtualModes_j = Omega->nodes_[i].virtualModes_;
            std::transform(operator_virtualModes_j.begin(), operator_virtualModes_j.end(), operator_virtualModes_j.begin(), 
                          [&currentMode, &currentConjPsiMode](int32_t x) -> int32_t {return x + currentMode + currentConjPsiMode;});
            operator_modes_j.insert(operator_modes_j.end(), operator_virtualModes_j.begin(), operator_virtualModes_j.end());

            size_t idx2 = 2UL * numprocessingNodes + j;

            modesIn[idx2] = operator_modes_j;
            extentsIn[idx2] = Omega->GetTensorExtents(i, false);

            tensorsIn[idx2] = Omega->tensors_[i];

            qualifiersIn[idx2].isConjugate = 0;
            qualifiersIn[idx2].isConstant = 1;
            qualifiersIn[idx2].requiresGradient = 0;
        }

        std::vector<int32_t> omegaModes_left;
        std::vector<int64_t> omegaExtents_left;
        std::vector<int32_t> omegaModes_right;
        std::vector<int64_t> omegaExtents_right;

        for(size_t j = numprocessingNodes; j < numSites_; ++j) 
        {
            const size_t& i = keep_nodes[j - numprocessingNodes];
           
            std::vector<int32_t> psi_modes_j = GetTensorModes(i, false);
            std::vector<int64_t> psi_extents_j = GetTensorExtents(i, false);
            
            omegaModes_left.insert(omegaModes_left.end(), psi_modes_j.begin(), psi_modes_j.end());
            omegaExtents_left.insert(omegaExtents_left.end(), psi_extents_j.begin(), psi_extents_j.end());

            std::vector<int32_t> conjpsi_modes_j = ConjPsiphysModes[j];

            std::vector<int32_t> conjpsi_virtualModes_j = (ConjPsi == nullptr) ? nodes_[i].virtualModes_ : ConjPsi->nodes_[i].virtualModes_;
            std::transform(conjpsi_virtualModes_j.begin(), conjpsi_virtualModes_j.end(), conjpsi_virtualModes_j.begin(), 
                          [&currentMode](int32_t x) -> int32_t {return x + currentMode;});
            conjpsi_modes_j.insert(conjpsi_modes_j.end(), conjpsi_virtualModes_j.begin(), conjpsi_virtualModes_j.end());

            omegaModes_right.insert(omegaModes_right.end(), conjpsi_modes_j.begin(), conjpsi_modes_j.end());
            
            if(ConjPsi == nullptr)
            {
                omegaExtents_right.insert(omegaExtents_right.end(), psi_extents_j.begin(), psi_extents_j.end());
            }
            else
            {
                std::vector<int64_t> conjpsi_extents_j = ConjPsi->GetTensorExtents(i, false);
                omegaExtents_right.insert(omegaExtents_right.end(), conjpsi_extents_j.begin(), conjpsi_extents_j.end());
            }
        }

        ConjPsiphysModes.clear();

        auto removeDuplicates = [](std::vector<int32_t>& modes, std::vector<int64_t>& extents) 
                                {
                                    std::vector<bool> to_remove(modes.size(), false);
                                    
                                    for(size_t i = 0; i < modes.size(); ++i) 
                                    {
                                        for(size_t j = i + 1; j < modes.size(); ++j) 
                                        {
                                            if (modes[i] == modes[j]) 
                                            {
                                                if(!to_remove[i]) to_remove[i] = true;
                                                if(!to_remove[j]) to_remove[j] = true;
                                            }
                                        }
                                    }
                                    
                                    for(int64_t i = static_cast<int64_t>(to_remove.size()) - 1L; i >= 0L; --i) 
                                    {
                                        if(to_remove[i]) 
                                        {
                                            modes.erase(modes.begin() + i);
                                            extents.erase(extents.begin() + i);
                                        }
                                    }
                                };

        removeDuplicates(omegaModes_left, omegaExtents_left);
        removeDuplicates(omegaModes_right, omegaExtents_right);
        
        modesIn.back() = omegaModes_left;
        extentsIn.back() = omegaExtents_left;

        tensorsIn.back() = nullptr;

        qualifiersIn.back().isConjugate = 0;
        qualifiersIn.back().isConstant = 1;
        qualifiersIn.back().requiresGradient = 0;

        size_t dimOut_right = static_cast<size_t>(std::accumulate(omegaExtents_right.begin(),
                                                                  omegaExtents_right.end(),
                                                                  1L,
                                                                  [](int64_t acc, const int64_t& current) {return acc * current;}));

        std::unordered_map<size_t, size_t> parent;

        std::function<size_t(size_t)> find = [&parent, &find](size_t x) -> size_t 
        {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        };
    
        for (const auto& pair : graph_) 
        {                
            auto it1 = std::find(keep_nodes.begin(), keep_nodes.end(), pair.first);
            auto it2 = std::find(keep_nodes.begin(), keep_nodes.end(), pair.second);

            if(it1 == keep_nodes.end())
            {
                if (!parent.count(pair.first)) 
                {
                    parent[pair.first] = pair.first;
                }
            }

            if(it2 == keep_nodes.end())
            {
                if (!parent.count(pair.second)) 
                {
                    parent[pair.second] = pair.second;
                }
            }

            if((it1 == keep_nodes.end()) && (it2 == keep_nodes.end()))
            {
                parent[find(pair.first)] = find(pair.second);
            }
        }

        std::unordered_map<size_t, std::set<size_t>> temp;
    
        for (const auto& p : parent) 
        {
            temp[find(p.first)].insert(p.first);
        }
    
        std::vector<std::set<size_t>> leaves;
    
        for (auto& group : temp) 
        {
            leaves.push_back(std::move(group.second));
        }

        temp.clear();
        parent.clear();

        size_t numLeaves = leaves.size();
        size_t numInputTensorsNew = numLeaves + numkeepNodes + 1UL;

        std::vector<std::vector<int32_t>> modesInNew(numInputTensorsNew);
        std::vector<std::vector<int64_t>> extentsInNew(numInputTensorsNew);
        std::vector<cutensornetTensorQualifiers_t> qualifiersInNew(numInputTensorsNew);
        std::vector<const void*> tensorsInNew(numInputTensorsNew);

        auto findFreeModes = [](const std::vector<std::vector<int32_t>>& in_modes,
                                const std::vector<std::vector<int64_t>>& in_extents,
                                std::vector<int32_t>& out_modes,
                                std::vector<int64_t>& out_extents)
                               {
                                    std::unordered_map<int32_t, int16_t> modes_count;
                                    std::unordered_map<int32_t, int64_t> modes_vs_extents;
                               
                                    const size_t num_nodes = in_modes.size();
                               
                                    for(size_t i = 0; i < num_nodes; ++i) 
                                    {
                                        const size_t num_modes = in_modes[i].size();
                                        const auto& modi = in_modes[i];
                                        const auto& exti = in_extents[i];
                                        
                                        for(size_t j = 0; j < num_modes; ++j) 
                                        {
                                            const int32_t modj = modi[j];
                                       
                                            modes_count[modj]++;                                      
                                            modes_vs_extents[modj] = exti[j];
                                        }
                                    }
                                   
                                    std::vector<std::pair<int32_t, int64_t>> free_pairs;
                                    for(const auto& [mod, count] : modes_count) 
                                    {
                                        if(count == 1) 
                                        {
                                            free_pairs.emplace_back(mod, modes_vs_extents[mod]);
                                        }
                                    }
                                   
                                    std::sort(free_pairs.begin(), free_pairs.end(),
                                              [](const auto& a, const auto& b) -> bool {return a.first < b.first;});
                                   
                                    for(const auto& [mod, ext] : free_pairs)
                                    {
                                        out_modes.push_back(mod);
                                        out_extents.push_back(ext);
                                    }
                               };

        for(size_t i = 0; i < numLeaves; ++i)
        {
            const auto& leave = leaves[i];

            size_t cl_id = cache.FindSubLeaveIndex(leave);

            std::set<size_t> cached_leave;
            if(cl_id != std::numeric_limits<size_t>::max())
            {
                cached_leave = cache.GetSites(cl_id);
            }

            std::set<size_t> diff_leave;
            std::set_difference(leave.begin(), leave.end(), 
                                cached_leave.begin(), cached_leave.end(), 
                                std::inserter(diff_leave, diff_leave.begin()));

            std::vector<std::vector<int32_t>> modesLeave; 
            std::vector<std::vector<int64_t>> extentsLeave;
            std::vector<cutensornetTensorQualifiers_t> qualifiersLeave;
            std::vector<const void*> tensorsLeave;

            for(const auto& node : diff_leave)
            {                   
                for(size_t g = 0; g < 3UL; ++g)
                {
                    modesLeave.push_back(modesIn[accessPoint[node] + g * numprocessingNodes]);
                    extentsLeave.push_back(extentsIn[accessPoint[node] + g * numprocessingNodes]);
                    qualifiersLeave.push_back(qualifiersIn[accessPoint[node] + g * numprocessingNodes]);
                    tensorsLeave.push_back(tensorsIn[accessPoint[node] + g * numprocessingNodes]);
                }  
            }

            if(!cached_leave.empty())
            {
                std::vector<std::vector<int32_t>> all_modes_cached_leave; 
                std::vector<std::vector<int64_t>> all_extents_cached_leave;

                for(const auto& node : cached_leave)
                {                   
                    for(size_t g = 0; g < 3UL; ++g)
                    {
                        all_modes_cached_leave.push_back(modesIn[accessPoint[node] + g * numprocessingNodes]);
                        all_extents_cached_leave.push_back(extentsIn[accessPoint[node] + g * numprocessingNodes]);;
                    }
                }

                std::vector<int32_t> modes_cached_leave;
                std::vector<int64_t> extents_cached_leave;

                findFreeModes(all_modes_cached_leave, all_extents_cached_leave, modes_cached_leave, extents_cached_leave);

                cutensornetTensorQualifiers_t qualifier_cached_leave;
                qualifier_cached_leave.isConjugate = 0;
                qualifier_cached_leave.isConstant = 1;
                qualifier_cached_leave.requiresGradient = 0;

                modesLeave.push_back(modes_cached_leave);
                extentsLeave.push_back(extents_cached_leave);
                qualifiersLeave.push_back(qualifier_cached_leave);
                tensorsLeave.push_back(cache.GetTensorData(cl_id));
            }

            std::vector<int32_t> out_modes;
            std::vector<int64_t> out_extents;

            findFreeModes(modesLeave, extentsLeave, out_modes, out_extents);

            size_t dimOut = static_cast<size_t>(std::accumulate(out_extents.begin(),
                                                                out_extents.end(),
                                                                1L,
                                                                [](int64_t acc, const int64_t& current) {return acc * current;}));

            void* tensorOut;
            HANDLE_CUDA_ERROR(cudaMalloc(&tensorOut, dimOut * sizeof(complexType)));

            CuTensorNetMethods::ContractTensors(handle_.at(thread_num),
                                                streams_.at(thread_num),
                                                modesLeave,
                                                extentsLeave,
                                                qualifiersLeave,
                                                tensorsLeave,
                                                out_modes,
                                                dimOut,
                                                tensorOut,
                                                0.8 / static_cast<double>(numStreams_),
                                                workSpaceLimit_,
                                                {},
                                                workSpacePreference_,
                                                optimizerAttributes,
                                                CuTensorNetMethods::MPI_);

            modesInNew[i] = out_modes;
            extentsInNew[i] = out_extents;

            qualifiersInNew[i].isConjugate = 0;
            qualifiersInNew[i].isConstant = 1;
            qualifiersInNew[i].requiresGradient = 0;

            tensorsInNew[i] = tensorOut;   

            cache.AddLeave(leave, out_modes, out_extents, tensorOut);
        }

        for(size_t i = 0; i < numkeepNodes; ++i)
        {
            modesInNew[i + numLeaves] = modesIn[i + 3UL * numprocessingNodes];
            extentsInNew[i + numLeaves] = extentsIn[i + 3UL * numprocessingNodes];
            qualifiersInNew[i + numLeaves] = qualifiersIn[i + 3UL * numprocessingNodes];
            tensorsInNew[i + numLeaves] = tensorsIn[i + 3UL * numprocessingNodes];
        }

        modesInNew.back() = modesIn.back();
        extentsInNew.back() = extentsIn.back();
        qualifiersInNew.back() = qualifiersIn.back();
        tensorsInNew.back() = tensorsIn.back();

        return {tensorsInNew, TensorNetDescriptor(modesInNew, 
                                                  extentsInNew, 
                                                  qualifiersInNew, 
                                                  omegaModes_right, 
                                                  dimOut_right)};
    }

    complexType TensorNetwork::FindGroundStateUsingDMRG(TensorNetwork* Psi,
                                                        double error_threshold,
                                                        size_t max_iter,
                                                        const std::vector<size_t>& max_virtual_extents,
                                                        size_t thread_num,
                                                        bool cached,
                                                        size_t verbose,
                                                        const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes,
                                                        int32_t numAutotuningIterations)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                         "The state of the current TN is not initialized.");
            }

            if(Psi->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                         "The state of Psi TN is not initialized.");
            }

            if(numSites_ != Psi->numSites_)
            {
                throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                         "The number of sites of the current TN differs from that of Psi TN.");
            }

            if(!Psi->loopFree_)
            {
                throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                         "The DMRG algorithm cannot be applied to loop TNs.");
            }

            if(graph_ != Psi->graph_)
            {
                throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                         "The graphs of the current TN and the Psi TN differ from each other.");
            }

            for(size_t i = 0; i < numSites_; ++i)
            {
                auto& siteI_node = this->nodes_[i];
                const auto& Psi_node = Psi->nodes_[i];
                
                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(!Psi_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of Psi TN.");
                }  

                if(siteI_node.physModes_.size() % 2UL != 0)
                {
                    throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                             "The current TN must be a operator.");
                }

                if(siteI_node.physModes_.size() / 2UL != Psi_node.physModes_.size())
                {
                    throw std::runtime_error("TensorNetwork::FindGroundStateUsingDMRG: "
                                             "The phys modes of the current TN and Psi TN are not suitable at site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        cutensornetTensorSVDPartition_t psi_partition;   

        HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigGetAttribute(Psi->handle_.at(thread_num), 
                                                                 Psi->svdConfig_.at(thread_num),
                                                                 CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION,
                                                                 &psi_partition,
                                                                 sizeof(psi_partition)));

        if(psi_partition != CUTENSORNET_TENSOR_SVD_PARTITION_SV)
        {
            cutensornetTensorSVDPartition_t local_partition = CUTENSORNET_TENSOR_SVD_PARTITION_SV;
        
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(Psi->handle_.at(thread_num), 
                                                                     Psi->svdConfig_.at(thread_num),
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, 
                                                                     &local_partition, 
                                                                     sizeof(local_partition)));
        }

        size_t orthogonalityCenter = std::numeric_limits<size_t>::max();

        auto [start_norm_device, start_descNorm] = Psi->GetDensityMatrix({}, 
                                                                         true, 
                                                                         thread_num, 
                                                                         optimizerAttributes);
        complexType start_norm_host;
        HANDLE_CUDA_ERROR(cudaMemcpy((void*)(&start_norm_host), start_norm_device, sizeof(complexType), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaFree(start_norm_device));

        graphTraversalType traversal = Psi->graphTraversalToRoot_;

        CachedLeaves cache;

        complexType previous_ground_value(1.0E150, 0.0);
        complexType new_ground_value(1.0E100, 0.0);

        size_t iter = 0;

        while(std::abs(previous_ground_value - new_ground_value) > error_threshold * std::abs(new_ground_value))
        {                 
            for(const auto& [siteA, siteB] : traversal)
            {                                                                  
                cache.EraseLeavesIf({siteA, siteB});
                
                if((siteA != orthogonalityCenter) && (siteB != orthogonalityCenter))
                {
                    std::set<size_t> changed_tensors = Psi->OrthogonalizeAround(siteB, orthogonalityCenter);

                    cache.EraseLeavesIf(changed_tensors);
                }

                void* tensorABStart = Psi->ComputeTwoSiteVector(siteA, siteB, false, thread_num);
                
                auto [rThisInTensors, rThisTNDescAB] = Psi->EvaluateTensorNetDescriptorOfEffectiveOperator(this, 
                                                                                                           cache,
                                                                                                           nullptr, 
                                                                                                           {siteA, siteB},
                                                                                                           thread_num,
                                                                                                           optimizerAttributes);

                OperatorVectorProduct* rThisbyVecProductAB = CuTensorNetMethods::BuildOperatorVectorProduct(Psi->handle_.at(thread_num),
                                                                                                            Psi->streams_.at(thread_num),
                                                                                                            rThisTNDescAB,
                                                                                                            rThisInTensors,
                                                                                                            0.8 / static_cast<double>(numStreams_),
                                                                                                            workSpaceLimit_,
                                                                                                            workSpacePreference_,
                                                                                                            optimizerAttributes,
                                                                                                            numAutotuningIterations,
                                                                                                            CuTensorNetMethods::MPI_);

                auto eignsys = CuOperatorMethods::HeOperatorGroundState(rThisbyVecProductAB->GetProductFunction(), 
                                                                        rThisbyVecProductAB->GetOutTensorSize(), 
                                                                        tensorABStart);
                
                HANDLE_CUDA_ERROR(cudaFree(tensorABStart));

                rThisbyVecProductAB->ClearTNMetaData();

                delete rThisbyVecProductAB;

                if(!cached)
                {
                    cache.Clear();
                }
                
                int64_t extentABbond = max_virtual_extents.empty() ? 0L :
                                       static_cast<int64_t>(max_virtual_extents[std::min({iter, max_virtual_extents.size() - 1UL})]);
                
                Psi->SetTwoSiteVector(siteA, 
                                      siteB, 
                                      eignsys.second, 
                                      extentABbond, 
                                      thread_num, 
                                      (verbose >= 2UL) ? true : false);

                orthogonalityCenter = siteB;

                HANDLE_CUDA_ERROR(cudaFree(eignsys.second));
            }

            auto [norm_device, descNorm] = Psi->GetDensityMatrix({}, 
                                                                 true, 
                                                                 thread_num, 
                                                                 optimizerAttributes);
                
            complexType norm_host;
            HANDLE_CUDA_ERROR(cudaMemcpy((void*)(&norm_host), norm_device, sizeof(complexType), cudaMemcpyDeviceToHost));
            HANDLE_CUDA_ERROR(cudaFree(norm_device));
            
            Psi->ApplyScalar(std::sqrt(start_norm_host) / std::sqrt(norm_host), {orthogonalityCenter});
                
            previous_ground_value = new_ground_value;

            new_ground_value = Psi->ComputeMatrixElement(this,
                                                         nullptr, 
                                                         thread_num, 
                                                         optimizerAttributes) / start_norm_host;

            ++iter;

            if(verbose >= 1UL)
            {
                std::cout << "TensorNetwork::FindGroundStateUsingDMRG: New ground value = " 
                          << std::scientific << std::setprecision(10) 
                          << new_ground_value << ";" << std::endl << std::defaultfloat;
            }

            if(iter >= max_iter)
            {
                std::cout << "TensorNetwork::FindGroundStateUsingDMRG: "
                             "The number of iterations has exceeded the limit." << std::endl;
                break;
            }

            std::reverse(traversal.begin(), traversal.end());
            for(auto& it : traversal)
            {
                std::swap(it.first, it.second);
            }
        }

        if(psi_partition != CUTENSORNET_TENSOR_SVD_PARTITION_SV)
        {
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(Psi->handle_.at(thread_num), 
                                                                     Psi->svdConfig_.at(thread_num),
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, 
                                                                     &psi_partition, 
                                                                     sizeof(psi_partition)));
        }
        
        return new_ground_value;
    }

    void TensorNetwork::UpdateUsingTDVP(const TensorNetwork* RHS,
                                        const Integrators::BaseIntegrator& solver,
                                        double dt,
                                        size_t edge,
                                        size_t order,
                                        bool cached,
                                        bool verbose,
                                        size_t thread_num,
                                        const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes,
                                        int32_t numAutotuningIterations)
    {
        if(check_)
        {
            if(tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                         "The state of the current TN is not initialized.");
            }

            if(RHS->tensors_.empty()) 
            {
                throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                         "The state of RHS TN is not initialized.");
            }

            if(numSites_ != RHS->numSites_)
            {
                throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                         "The number of sites of the current TN differs from that of RHS TN.");
            }

            if(!loopFree_)
            {
                throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                         "The TDVP algorithm cannot be applied to loop TNs.");
            }

            if(graph_ != RHS->graph_)
            {
                throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                         "The graphs of the current TN and RHS TN differ from each other.");
            }

            for(size_t i = 0; i < numSites_; ++i)
            {
                auto& siteI_node = this->nodes_[i];
                const auto& RHS_node = RHS->nodes_[i];
                
                if(!siteI_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of the current TN.");
                }

                if(!RHS_node.extra_virtualModes_.empty())
                {
                    throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                             "Duplicate bonds should be excluded for site " 
                                             + std::to_string(i) + " of RHS TN.");
                }  
                
                if(RHS_node.physModes_.size() / 2UL != siteI_node.physModes_.size())
                {
                    throw std::runtime_error("TensorNetwork::UpdateUsingTDVP: "
                                             "The phys modes of the current TN and RHS TN are not suitable at site " 
                                             + std::to_string(i) + ".");
                }
            }
        }

        cutensornetTensorSVDPartition_t psi_partition;   

        HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigGetAttribute(handle_.at(thread_num), 
                                                                 svdConfig_.at(thread_num),
                                                                 CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION,
                                                                 &psi_partition,
                                                                 sizeof(psi_partition)));

        if(psi_partition != CUTENSORNET_TENSOR_SVD_PARTITION_SV)
        {
            cutensornetTensorSVDPartition_t local_partition = CUTENSORNET_TENSOR_SVD_PARTITION_SV;
        
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_.at(thread_num), 
                                                                     svdConfig_.at(thread_num),
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, 
                                                                     &local_partition, 
                                                                     sizeof(local_partition)));
        }

        size_t orthogonalityCenter = std::numeric_limits<size_t>::max();

        bool loopFree;
        graphTraversalType traversal = GetGraphTraversalToRoot(graph_, numSites_, edge, loopFree);

        std::vector<size_t> visits(numSites_, 0UL);
        for(const auto& [node1, node2] : traversal)
        {
            ++visits[node1];
            ++visits[node2];
        }

        std::vector<size_t> evaluated(numSites_, 0UL);

        size_t num_edges = traversal.size();

        std::vector<std::vector<size_t>> back(num_edges - 1UL);

        std::vector<double> pk_vec = GetSuzukiCoeffs(order);

        CachedLeaves cache;

        for(size_t i = 0UL; i < pk_vec.size(); ++i)
        {
            const auto& pk = pk_vec[i];
            
            for(size_t sweep = 0UL; sweep < 2UL; ++sweep)
            {            
                for(size_t j = 0UL; j < num_edges; ++j)
                {
                    const auto& [siteA, siteB] = traversal[j];

                    cache.EraseLeavesIf({siteA, siteB});

                    if((siteA != orthogonalityCenter) && (siteB != orthogonalityCenter))
                    {
                        std::set<size_t> changed_tensors = OrthogonalizeAround(siteB, orthogonalityCenter);

                        cache.EraseLeavesIf(changed_tensors);
                    }

                    void* tensorInAB = ComputeTwoSiteVector(siteA, siteB, false, thread_num);

                    auto [rRHSInTensorsAB, rRHSTNDescAB] = EvaluateTensorNetDescriptorOfEffectiveOperator(RHS, 
                                                                                                          cache,
                                                                                                          nullptr, 
                                                                                                          {siteA, siteB},
                                                                                                          thread_num,
                                                                                                          optimizerAttributes);

                    OperatorVectorProduct* rRHSbyVecProductAB = CuTensorNetMethods::BuildOperatorVectorProduct(handle_.at(thread_num),
                                                                                                               streams_.at(thread_num),
                                                                                                               rRHSTNDescAB,
                                                                                                               rRHSInTensorsAB,
                                                                                                               0.8 / static_cast<double>(numStreams_),
                                                                                                               workSpaceLimit_,
                                                                                                               workSpacePreference_,
                                                                                                               optimizerAttributes,
                                                                                                               numAutotuningIterations,
                                                                                                               CuTensorNetMethods::MPI_);

                    void* tensorOutAB = solver.Integrate(rRHSbyVecProductAB->GetProductFunction(), 
                                                         rRHSbyVecProductAB->GetOutTensorSize(), 
                                                         dt * pk / 2.0, 
                                                         tensorInAB);
                    
                    HANDLE_CUDA_ERROR(cudaFree(tensorInAB));

                    rRHSbyVecProductAB->ClearTNMetaData();

                    delete rRHSbyVecProductAB;

                    if(!cached)
                    {
                        cache.Clear();
                    }

                    SetTwoSiteVector(siteA, 
                                     siteB, 
                                     tensorOutAB, 
                                     0UL, 
                                     thread_num, 
                                     verbose);

                    orthogonalityCenter = siteB;

                    HANDLE_CUDA_ERROR(cudaFree(tensorOutAB));

                    if((i == 0UL) && (sweep == 0UL))
                    {
                        ++evaluated[siteA];
                        ++evaluated[siteB];
                        
                        for(const auto& site : {siteA, siteB})
                        {
                            if(evaluated[site] < visits[site])
                            {
                                back.at(j).push_back(site);
                            }
                        }   
                    }

                    if(j < num_edges - 1UL)
                    {
                        for(const auto& site : back[j])
                        {                            
                            cache.EraseLeavesIf({site});
                            
                            if(site != orthogonalityCenter)
                            {
                                std::set<size_t> changed_tensors = OrthogonalizeAround(site, orthogonalityCenter);

                                cache.EraseLeavesIf(changed_tensors);

                                orthogonalityCenter = site;
                            }

                            auto [rRHSInTensorsS, rRHSTNDescS] = EvaluateTensorNetDescriptorOfEffectiveOperator(RHS, 
                                                                                                                cache,
                                                                                                                nullptr, 
                                                                                                                {site},
                                                                                                                thread_num,
                                                                                                                optimizerAttributes);

                            OperatorVectorProduct* rRHSbyVecProductS = CuTensorNetMethods::BuildOperatorVectorProduct(handle_.at(thread_num),
                                                                                                                      streams_.at(thread_num),
                                                                                                                      rRHSTNDescS,
                                                                                                                      rRHSInTensorsS,
                                                                                                                      0.8 / static_cast<double>(numStreams_),
                                                                                                                      workSpaceLimit_,
                                                                                                                      workSpacePreference_,
                                                                                                                      optimizerAttributes,
                                                                                                                      numAutotuningIterations,
                                                                                                                      CuTensorNetMethods::MPI_);
                            
                            void* tensorOutS = solver.Integrate(rRHSbyVecProductS->GetProductFunction(), 
                                                                rRHSbyVecProductS->GetOutTensorSize(), 
                                                                -dt * pk / 2.0, 
                                                                tensors_[site]);
                            
                            HANDLE_CUDA_ERROR(cudaFree(tensors_[site]));

                            tensors_[site] = tensorOutS;

                            rRHSbyVecProductS->ClearTNMetaData();

                            delete rRHSbyVecProductS;

                            if(!cached)
                            {
                                cache.Clear();
                            }
                        }
                    }
                }

                std::reverse(traversal.begin(), traversal.end());
                for(auto& it : traversal)
                {
                    std::swap(it.first, it.second);
                }

                std::reverse(back.begin(), back.end());
                for(auto& it : back)
                {
                    std::reverse(it.begin(), it.end());
                }
            }
        }

        if(psi_partition != CUTENSORNET_TENSOR_SVD_PARTITION_SV)
        {
            HANDLE_CUTN_ERROR(cutensornetTensorSVDConfigSetAttribute(handle_.at(thread_num), 
                                                                     svdConfig_.at(thread_num),
                                                                     CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION, 
                                                                     &psi_partition, 
                                                                     sizeof(psi_partition)));
        }
    }
}