#pragma once

#include <cmath>
#include <cstring> 
#include <cassert>

#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <stack>
#include <map>
#include <queue>
#include <unordered_set>
#include <tuple>
#include <unordered_map>
#include <algorithm>
#include <complex>
#include <numeric>
#include <utility>
#include <thread>
#include <stdexcept>
#include <filesystem>
#include <fstream>

#include <yaml-cpp/yaml.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cutensornet.h>

#include "CuErrorUtils.hh"
#include "CuOperatorMethods.hh"
#include "TensorDescriptor.hh"
#include "TensorNetDescriptor.hh"
#include "OperatorVectorProduct.hh"
#include "BaseIntegrator.hh"
#include "CuTensorNetMethods.hh"
#include "CachedLeaves.hh"

namespace QTensorNet
{
    static_assert(sizeof(size_t) == sizeof(uint64_t), 
                  "Architecture not supported: TensorNetwork requires 64-bit size_t.");
    
    struct Node
    {        
        std::vector<int32_t> physModes_;
        std::vector<int64_t> physExtents_;

        std::vector<int32_t> virtualModes_;
        std::vector<int64_t> virtualExtents_;
        std::unordered_map<size_t, size_t> neighbors_;
        
        std::vector<int32_t> extra_virtualModes_;
        std::vector<int64_t> extra_virtualExtents_;
    };
    
    struct TupleComparator 
    {
        template <typename Tuple>
        bool operator()(const Tuple& lhs, const Tuple& rhs) const 
        {
            auto [l1, l2] = std::minmax(std::get<0>(lhs), std::get<1>(lhs));
            auto [r1, r2] = std::minmax(std::get<0>(rhs), std::get<1>(rhs));

            return std::tie(l1, l2) < std::tie(r1, r2);
        }
    };

    using complexType = std::complex<double>;
    using graphType = std::set<std::pair<size_t, size_t>>;
    using virtualModesGraphType = std::set<std::tuple<size_t, size_t, int64_t>, TupleComparator>;
    using graphTraversalType = std::vector<std::pair<size_t, size_t>>;
    using SingleOp = std::tuple<size_t, complexType, std::vector<complexType>>;
    using OpTerm = std::vector<SingleOp>;

    std::vector<double> GetSuzukiCoeffs(size_t k);
    graphTraversalType GetGraphTraversalToRoot(const graphType& graph, 
                                               size_t numSites, 
                                               size_t root, 
                                               bool& loop_free);
    void BuildOpTensors(virtualModesGraphType& graph, 
                        size_t numSites, 
                        size_t root, 
                        const std::vector<OpTerm>& OpTerms, 
                        std::vector<std::vector<complexType>>& tensors_host);

    class TensorNetwork
    {
    public:
        TensorNetwork(const std::vector<std::vector<int64_t>>& physExtents,
                      const virtualModesGraphType& graph, size_t root = 0UL, int64_t maxVirtualExtent = 1L, 
                      size_t numStreams = 1UL, size_t workSpaceLimit = 4096UL /*MBytes*/);
        TensorNetwork(const std::string& path);
        ~TensorNetwork();
        TensorNetwork(const TensorNetwork& other);
        TensorNetwork(TensorNetwork&& other) noexcept;
        TensorNetwork& operator=(const TensorNetwork& other);
        TensorNetwork& operator=(TensorNetwork&& other) noexcept;
        void Save(const std::string& path);
        void Load(const std::string& path);
        void SetSVDConfig(double absCutoff, 
                          double relCutoff, 
                          cutensornetTensorSVDPartition_t partition = CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL,
                          cutensornetTensorSVDNormalization_t renorm = CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE,
                          cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD,
                          const void* svdParams = nullptr, size_t svdParamsSize = 0);
        void SetMaxVirtualExtent(int64_t val);
        void SetWorkSpacePreference(cutensornetWorksizePref_t pref);
        void SetWorkSpaceLimit(size_t val /*MBytes*/);
        void SetGlobalMode(bool mode);
        const Node& GetNode(size_t id) const;
        Node& GetNode(size_t id);
        size_t GetNumSites() const;
        size_t GetTensorSize(size_t id, bool include_extra = true) const;
        std::vector<int32_t> GetTensorModes(size_t id, bool include_extra = true) const;
        std::vector<int64_t> GetTensorExtents(size_t id, bool include_extra = true) const;
        void SetState(const std::vector<std::vector<complexType>>& tensors_host);
        void ClearTensors();
        void ClearNet();
        int32_t GetNextMode(bool update = true);
        cutensornetHandle_t GetHandle(size_t id);
        cudaStream_t GetStream(size_t id);
        std::vector<complexType> GetTensorData(size_t id) const;
        void SetTensorData(size_t id, const std::vector<complexType>& ten);
        const void* GetTensor(size_t id) const;
        void SetTensor(size_t id, void* ten, size_t ten_size);
        void SynchronizeStreams(const std::vector<size_t>& ids = {});
        void* ComputeTwoSiteVector(size_t siteA, 
                                   size_t siteB,
                                   bool conjugate = false,
                                   size_t thread_num = 0UL);
        void SetTwoSiteVector(size_t siteA, 
                              size_t siteB,
                              const void* tensorInAB,
                              int64_t maxVirtualExtent = 0L,
                              size_t thread_num = 0UL,
                              bool verbose = false);
        std::set<size_t> OrthogonalizeAround(size_t center_id, 
                                             size_t pre_center_id = std::numeric_limits<size_t>::max());
        void Shrink(bool verbose = false,
                    size_t thread_num = 0UL);
        void ApplySingleSiteGate(size_t site, 
                                 const void* operatorData,
                                 std::vector<int32_t> operatorModes,
                                 std::vector<int64_t> operatorExtents,
                                 size_t thread_num = 0UL);
        void ApplyTwoSiteGate(size_t siteA, 
                              size_t siteB, 
                              const void* operatorData,
                              std::vector<int32_t> operatorModes,
                              std::vector<int64_t> operatorExtents,
                              size_t thread_num = 0UL,
                              bool verbose = false);
        TensorNetwork& Conjugate();
        TensorNetwork& Transpose();
        void ApplyScalar(complexType scalar, const std::vector<size_t>& nodes = {});
        TensorNetwork& operator*=(complexType scalar);
        TensorNetwork operator*(complexType scalar) const;
        friend TensorNetwork operator*(complexType scalar, const TensorNetwork& Psi);
        void ApplyTensorNetOperator(const TensorNetwork* Omega, 
                                    bool verbose = false);
        TensorNetwork& operator*=(const TensorNetwork& Psi);
        TensorNetwork operator*(const TensorNetwork& Psi) const;
        void AddTensorNet(const TensorNetwork* Psi, 
                          bool verbose = false);
        TensorNetwork& operator+=(const TensorNetwork& Psi);
        TensorNetwork operator+(const TensorNetwork& Psi) const;
        TensorNetwork* EvaluateTensorNetProduct(const TensorNetwork* Omega, 
                                                const std::pair<bool, bool>& transpose = {false, false},
                                                const std::pair<bool, bool>& conjugate = {false, false},
                                                int64_t maxVirtualExtent = 0L,
                                                bool verbose = false) const;
        complexType ComputeMatrixElement(const void* operatorData,
                                         std::vector<int32_t> operatorModes,
                                         std::vector<int64_t> operatorExtents,
                                         const TensorNetwork* ConjPsi = nullptr,
                                         size_t thread_num = 0UL,
                                         const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes);
        complexType ComputeMatrixElement(const TensorNetwork* Omega,
                                         const TensorNetwork* ConjPsi = nullptr,
                                         size_t thread_num = 0UL,
                                         const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes);
        complexType ComputeOperatorTrace(const void* operatorData,
                                         std::vector<int32_t> operatorModes,
                                         std::vector<int64_t> operatorExtents,
                                         size_t thread_num = 0UL,
                                         const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes);
        complexType ComputeOperatorTrace(const TensorNetwork* Omega = nullptr,
                                         size_t thread_num = 0UL,
                                         const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes);
        void ExcludeExtraBond(size_t siteA, 
                              size_t siteB,
                              size_t thread_num = 0UL,
                              bool verbose = false);
        std::pair<void*, TensorDescriptor> GetDensityMatrix(const std::vector<size_t>& keep_nodes = {},
                                                            bool circuit_order = true,
                                                            size_t thread_num = 0UL,
                                                            const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes);
        std::pair<std::vector<const void*>, TensorNetDescriptor> EvaluateTensorNetDescriptorOfEffectiveOperator(const TensorNetwork* Omega,
                                                                                                                CachedLeaves& cache,
                                                                                                                const TensorNetwork* ConjPsi = nullptr,
                                                                                                                const std::vector<size_t>& keep_nodes = {},
                                                                                                                size_t thread_num = 0UL,
                                                                                                                const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes);
        complexType FindGroundStateUsingDMRG(TensorNetwork* Psi,
                                             double error_threshold = 1.0E-5,
                                             size_t max_iter = 10UL,
                                             const std::vector<size_t>& max_virtual_extents = {},
                                             size_t thread_num = 0UL,
                                             bool cached = true,
                                             size_t verbose = 0UL,
                                             const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes,
                                             int32_t numAutotuningIterations = 0);
        void UpdateUsingTDVP(const TensorNetwork* RHS,
                             const Integrators::BaseIntegrator& solver,
                             double dt,
                             size_t edge = 0UL,
                             size_t order = 1UL,
                             bool cached = true,
                             bool verbose = false,
                             size_t thread_num = 0UL,
                             const CuTensorNetMethods::ContractionOptimizerAttributes& optimizerAttributes = CuTensorNetMethods::optimalContractionOptimizerAttributes,
                             int32_t numAutotuningIterations = 0);

        static bool check_;

    private:
        bool loopFree_;
        bool globalMode_;
        std::vector<Node> nodes_;
        graphType graph_;
        graphTraversalType graphTraversalToRoot_;
        size_t numSites_;
        int64_t maxVirtualExtent_;
        std::vector<void*> tensors_;
        std::vector<cutensornetHandle_t> handle_;
        std::vector<cudaStream_t> streams_;
        std::vector<cutensornetTensorSVDConfig_t> svdConfig_;
        int32_t nextMode_{0};
        size_t numStreams_;
        cutensornetWorksizePref_t workSpacePreference_{CUTENSORNET_WORKSIZE_PREF_MIN};
        size_t workSpaceLimit_;
        double absCutoff_{0};
        double relCutoff_{0}; 
        cutensornetTensorSVDPartition_t partition_{CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL};
        cutensornetTensorSVDNormalization_t renorm_{CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE};
        cutensornetTensorSVDAlgo_t svdAlgo_{CUTENSORNET_TENSOR_SVD_ALGO_GESVD};
        void* svdParams_{nullptr};
        size_t svdParamsSize_{0};
    };
}