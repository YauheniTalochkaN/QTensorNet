#include <fstream>
#include <chrono>
#include <functional>
#include <random>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"
#include "ThreadPool.hh"

void DoTask(QTensorNet::ThreadPool& pool, 
            const std::function<void(const QTensorNet::TensorNetwork&, size_t, size_t, size_t)>& task, 
            const QTensorNet::TensorNetwork& ttn,
            size_t stream_num, size_t num_sites)
{
    size_t chunk_size = (num_sites + stream_num - 1) / stream_num;

    std::vector<std::future<void>> futures;

    for(size_t st = 0; st < stream_num; ++st)
    {
        size_t start = st * chunk_size;
        size_t end = std::min(start + chunk_size, num_sites);

        if(start < num_sites)
        {
            futures.emplace_back(pool.AddTask(task, ttn, st, start, end));
        }
        else
        {
            break;
        }
    }

    for(auto& future : futures)
    {
        future.get();
    }
};

int main(int argc, char* argv[])
{
    HANDLE_MPI_ERROR(MPI_Init(&argc, &argv));

    int rank{-1};
    HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int numProcs{0};
    HANDLE_MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &numProcs));
    
    if(rank == 0)
    {
        const size_t cuTensornetVersion = cutensornetGetVersion();
        std::cout << "cuTensorNet-vers: " << cuTensornetVersion << std::endl;
    }

    int numDevices{0};
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));

    const int deviceId = rank % numDevices;
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));

    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    std::cout << "\nRank: " << rank << "\n"
              << "GPU-local-id:" << deviceId << "\n"
              << "GPU-name:" << prop.name << std::endl;

    //QTensorNet::TensorNetwork::check_ = false;

    if(numProcs > 1)
    {
        QTensorNet::CuTensorNetMethods::MPI_ = true;
    }

    if(rank == 0)
    {
        auto start = std::chrono::steady_clock::now();

        size_t numSites = 60UL;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtentTTS = 800L;
        int64_t maxVirtualExtentTTO = 200L;
        double absCutoffTTS = 0.0;
        double absCutoffTTO = 0.0;
        double relCutoffTTS = 1.0e-8;
        double relCutoffTTO = 1.0e-8;
        size_t workSpaceLimitTTS = 50UL * 1024UL;
        size_t workSpaceLimitTTO = 50UL * 1024UL;

        size_t num_threads = 10UL;
        QTensorNet::ThreadPool pool(num_threads);

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 100},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 5000}};

        std::vector<size_t> max_virtual_extents = {20UL, 20UL, 
                                                   30UL, 30UL, 
                                                   50UL, 50UL, 
                                                   100UL, 100UL, 
                                                   200UL, 200UL, 
                                                   400UL, 400UL,
                                                   800UL, 800UL};
    
        size_t rootTTS = 54UL;
        size_t rootTTO = 54UL;

        QTensorNet::complexType J(-1.0, 0.0);
        QTensorNet::complexType dJ(-1.0, 0.0);

        std::vector<double> phi_list = {2.0 * M_PI / 3.0, -2.0 * M_PI / 3.0, 0.0};

        /* std::vector<std::tuple<size_t, size_t, size_t>> latt = {{1, 3, 0},   {2, 6, 0},   {7, 9, 0},   {5, 10, 0},  {8, 12, 0},  {11, 15, 0}, 
                                                                {13, 17, 0}, {14, 18, 0}, {16, 20, 0}, {19, 22, 0}, {0, 21, 0},  {4, 23, 0}, 
                                                                {0, 1, 1},   {4, 2, 1},   {3, 7, 1},   {6, 8, 1},   {10, 11, 1}, {12, 13, 1}, 
                                                                {15, 16, 1}, {18, 19, 1}, {20, 21, 1}, {22, 23, 1}, {5, 9, 1},   {14, 17, 1}, 
                                                                {0, 2, 2},   {4, 5, 2},   {3, 8, 2},   {6, 11, 2},  {9, 13, 2},  {10, 14, 2}, 
                                                                {12, 16, 2}, {15, 19, 2}, {17, 21, 2}, {20, 23, 2}, {1, 18, 2},  {7, 22, 2}}; */

        /* std::vector<std::tuple<size_t, size_t, size_t>> latt = {{0, 2, 0}, {4, 6, 0}, {3, 9, 0}, {7, 11, 0}, {8, 12, 0}, 
                                                                {10, 14, 0}, {13, 17, 0}, {15, 19, 0}, {16, 20, 0}, {18, 22, 0}, 
                                                                {21, 25, 0}, {23, 27, 0}, {24, 28, 0}, {26, 30, 0}, {5, 31, 0}, 
                                                                {1, 29, 0}, {1, 3, 1}, {5, 7, 1}, {2, 8, 1}, {6, 10, 1}, 
                                                                {9, 13, 1}, {11, 15, 1}, {12, 16, 1}, {14, 18, 1}, {17, 21, 1}, 
                                                                {19, 23, 1}, {20, 24, 1}, {22, 26, 1}, {25, 29, 1}, {27, 31, 1}, 
                                                                {4, 30, 1}, {0, 28, 1}, {0, 1, 2}, {4, 5, 2}, {3, 6, 2}, 
                                                                {8, 9, 2}, {10, 11, 2}, {13, 14, 2}, {16, 17, 2}, {18, 19, 2}, 
                                                                {21, 22, 2}, {24, 25, 2}, {26, 27, 2}, {29, 30, 2}}; */

        std::vector<std::tuple<size_t, size_t, size_t>> latt = {{1, 3, 0}, {5, 7, 0}, {9, 11, 0}, {2, 12, 0}, {6, 14, 0}, {10, 16, 0}, 
                                                                {13, 19, 0}, {15, 21, 0}, {17, 23, 0}, {18, 24, 0}, {20, 26, 0}, 
                                                                {22, 28, 0}, {25, 31, 0}, {27, 33, 0}, {29, 35, 0}, {30, 36, 0}, 
                                                                {32, 38, 0}, {34, 40, 0}, {37, 43, 0}, {39, 45, 0}, {41, 47, 0}, 
                                                                {42, 48, 0}, {44, 50, 0}, {46, 52, 0}, {49, 55, 0}, {51, 57, 0}, 
                                                                {53, 59, 0}, {0, 54, 0}, {4, 56, 0}, {8, 58, 0}, {0, 2, 1}, 
                                                                {4, 6, 1}, {8, 10, 1}, {3, 13, 1}, {7, 15, 1}, {11, 17, 1}, {12, 18, 1}, 
                                                                {14, 20, 1}, {16, 22, 1}, {19, 25, 1}, {21, 27, 1}, {23, 29, 1}, 
                                                                {24, 30, 1}, {26, 32, 1}, {28, 34, 1}, {31, 37, 1}, {33, 39, 1}, 
                                                                {35, 41, 1}, {36, 42, 1}, {38, 44, 1}, {40, 46, 1}, {43, 49, 1}, 
                                                                {45, 51, 1}, {47, 53, 1}, {48, 54, 1}, {50, 56, 1}, {52, 58, 1}, 
                                                                {1, 55, 1}, {5, 57, 1}, {9, 59, 1}, {0, 1, 2}, {4, 5, 2}, {8, 9, 2}, 
                                                                {3, 6, 2}, {7, 10, 2}, {12, 13, 2}, {14, 15, 2}, {16, 17, 2}, {19, 20, 2}, 
                                                                {21, 22, 2}, {24, 25, 2}, {26, 27, 2}, {28, 29, 2}, {31, 32, 2}, 
                                                                {33, 34, 2}, {36, 37, 2}, {38, 39, 2}, {40, 41,  2}, {43, 44, 2}, {45, 46, 2}, 
                                                                {48, 49, 2}, {50, 51, 2}, {52, 53, 2}, {55, 56, 2}, {57, 58, 2}};

        /* QTensorNet::virtualModesGraphType graph = {{1, 3, 1},   {2, 6, 1},   {5, 10, 1},  {11, 15, 1}, {13, 17, 1}, 
                                                   {16, 20, 1}, {19, 22, 1}, {4, 2, 1},   {3, 7, 1},   {6, 8, 1}, 
                                                   {10, 11, 1}, {12, 13, 1}, {15, 16, 1}, {18, 19, 1}, {20, 21, 1}, 
                                                   {0, 2, 1},   {3, 8, 1},   {6, 11, 1},  {9, 13, 1},  {10, 14, 1}, 
                                                   {12, 16, 1}, {15, 19, 1}, {20, 23, 1}}; */

        /* QTensorNet::virtualModesGraphType graph = {{0, 2, 1}, {4, 6, 1}, {3, 9, 1}, {7, 11, 1}, {8, 12, 1}, 
                                                   {10, 14, 1}, {13, 17, 1}, {16, 20, 1}, {18, 22, 1}, {21, 25, 1},
                                                   {24, 28, 1}, {26, 30, 1}, {1, 3, 1}, {5, 7, 1}, {2, 8, 1}, 
                                                   {6, 10, 1}, {9, 13, 1}, {11, 15, 1}, {14, 18, 1}, {17, 21, 1}, 
                                                   {19, 23, 1}, {22, 26, 1}, {25, 29, 1}, {27, 31, 1}, {0, 1, 1}, 
                                                   {4, 5, 1}, {16, 17, 1}, {18, 19, 1}, {24, 25, 1}, {26, 27, 1}, {29, 30, 1}}; */

        QTensorNet::virtualModesGraphType graph = {{1, 3, 1}, {5, 7, 1}, {9, 11, 1}, {2, 12, 1}, {6, 14, 1}, {10, 16, 1}, 
                                                   {13, 19, 1}, {15, 21, 1}, {17, 23, 1}, {18, 24, 1}, {20, 26, 1}, 
                                                   {22, 28, 1}, {25, 31, 1}, {27, 33, 1}, {29, 35, 1}, {30, 36, 1}, 
                                                   {32, 38, 1}, {34, 40, 1}, {37, 43, 1}, {39, 45, 1}, {41, 47, 1}, 
                                                   {42, 48, 1}, {44, 50, 1}, {46, 52, 1}, {49, 55, 1}, {51, 57, 1}, 
                                                   {53, 59, 1}, {0, 2, 1}, {4, 6, 1}, {8, 10, 1}, {3, 13, 1}, {7, 15, 1}, 
                                                   {11, 17, 1}, {12, 18, 1}, {14, 20, 1}, {16, 22, 1}, {19, 25, 1}, 
                                                   {21, 27, 1}, {23, 29, 1}, {24, 30, 1}, {26, 32, 1}, {28, 34, 1}, 
                                                   {31, 37, 1}, {33, 39, 1}, {35, 41, 1}, {36, 42, 1}, {38, 44, 1}, 
                                                   {40, 46, 1}, {43, 49, 1}, {45, 51, 1}, {47, 53, 1}, {48, 54, 1}, 
                                                   {50, 56, 1}, {52, 58, 1}, {0, 1, 1}, {4, 5, 1}, {8, 9, 1}, {55, 56, 1}, {57, 58, 1}};

        std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

        QTensorNet::TensorNetwork init_tts(physExtentsVec, graph, rootTTS, maxVirtualExtentTTS, 1UL, workSpaceLimitTTS);

        std::vector<std::vector<QTensorNet::complexType>> tts_tensors_host;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        std::cout << "\nInitializing TTS tensors for initial state..." << std::endl;

        for (size_t i = 0UL; i < numSites; ++i)
        {        
            std::vector<QTensorNet::complexType> data_host(init_tts.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

            data_host[0] = QTensorNet::complexType(dist(gen), dist(gen));
            data_host[1] = QTensorNet::complexType(dist(gen), dist(gen));

            tts_tensors_host.push_back(data_host);

            std::cout << "Site " << i << ": tensor[0] = (" << tts_tensors_host[i][0].real() 
                      << ", " << tts_tensors_host[i][0].imag() << "), tensor[1] = (" 
                      << tts_tensors_host[i][1].real() << ", " << tts_tensors_host[i][1].imag() << ")" << std::endl;
        }

        try
        {
            init_tts.SetState(tts_tensors_host);
            init_tts.SetSVDConfig(absCutoffTTS, relCutoffTTS);

            auto [norm_device, descNorm] = init_tts.GetDensityMatrix({}, true, 0UL, optimizer_attributes);
            auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

            HANDLE_CUDA_ERROR(cudaFree(norm_device));

            init_tts *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }

        //---------------------------------------------------------------------------------

        auto Sx_host = QTensorNet::BasisGates::SigmaX(0.5);
        void* Sx_device = QTensorNet::CuArrayMethods::VectorToGPUArray(Sx_host);
        
        auto Sy_host = QTensorNet::BasisGates::SigmaY(0.5);
        void* Sy_device = QTensorNet::CuArrayMethods::VectorToGPUArray(Sy_host);
        
        auto Sz_host = QTensorNet::BasisGates::SigmaZ(0.5);
        void* Sz_device = QTensorNet::CuArrayMethods::VectorToGPUArray(Sz_host);
        
        auto SS_host = QTensorNet::BasisGates::SigmaISigmaJSum(0.25, 0.25, 0.25);
        void* SS_device = QTensorNet::CuArrayMethods::VectorToGPUArray(SS_host);

        //---lambda functions--------------------------------------------------------------

        std::vector<std::tuple<QTensorNet::complexType, 
                               QTensorNet::complexType, 
                               QTensorNet::complexType>> S_obs(numSites);

        std::vector<std::pair<size_t, size_t>> pairs;
        
        for(size_t i = 0UL; i < numSites; ++i)
        {
            for(size_t j = i + 1UL; j < numSites; ++j)
            {
                pairs.emplace_back(i, j);
            }
        }

        std::vector<QTensorNet::complexType> SS_obs(pairs.size());

        auto thread_func_S_obs = [Sx_device, Sy_device, Sz_device, &S_obs, &optimizer_attributes]
                                 (const QTensorNet::TensorNetwork& tts, size_t stream_num, size_t start, size_t end) 
        {
            for(size_t k = start; k < end; ++k)
            {
                try
                {
                    std::vector<int32_t> SkModes = {tts.GetNode(k).physModes_[0], 
                                                    tts.GetNode(k).physModes_[0]};
                    std::vector<int64_t> SkExtents = {2, 2};

                    QTensorNet::complexType Sx = tts.ComputeMatrixElement(Sx_device, 
                                                                          SkModes, 
                                                                          SkExtents,
                                                                          nullptr,
                                                                          stream_num, 
                                                                          optimizer_attributes);

                    QTensorNet::complexType Sy = tts.ComputeMatrixElement(Sy_device, 
                                                                          SkModes, 
                                                                          SkExtents,
                                                                          nullptr,
                                                                          stream_num, 
                                                                          optimizer_attributes);
                    
                    QTensorNet::complexType Sz = tts.ComputeMatrixElement(Sz_device, 
                                                                          SkModes, 
                                                                          SkExtents,
                                                                          nullptr,
                                                                          stream_num, 
                                                                          optimizer_attributes);

                    S_obs.at(k) = {Sx, Sy, Sz};
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }
            }
        };

        auto thread_func_SS_obs = [SS_device, &SS_obs, &pairs, &optimizer_attributes]
                                  (const QTensorNet::TensorNetwork& tts, size_t stream_num, size_t start, size_t end) 
        {
            for(size_t k = start; k < end; ++k)
            {
                try
                {
                    auto [i, j] = pairs.at(k);
                    
                    std::vector<int32_t> SiSjModes = {tts.GetNode(i).physModes_[0],
                                                      tts.GetNode(j).physModes_[0], 
                                                      tts.GetNode(i).physModes_[0],
                                                      tts.GetNode(j).physModes_[0]};
                    std::vector<int64_t> SiSjExtents = {2, 2, 2, 2};
                    
                    SS_obs.at(k) = tts.ComputeMatrixElement(SS_device, 
                                                            SiSjModes, 
                                                            SiSjExtents,
                                                            nullptr,
                                                            stream_num, 
                                                            optimizer_attributes);

                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }
            }
        };

        //---------------------------------------------------------------------------------

        std::cout << "\nLooking for the ground TTS vector of the hamiltonian..." << std::endl;

        std::vector<QTensorNet::complexType> Jpmpm_list(41);

        for(size_t i = 0UL; i < Jpmpm_list.size(); ++i)
        {
            Jpmpm_list[i] = -0.6 + 1.2 * static_cast<double>(i) / static_cast<double>(Jpmpm_list.size() - 1UL);
        }

        std::vector<QTensorNet::complexType> Jzpm_list(41);
        
        for(size_t i = 0UL; i < Jzpm_list.size(); ++i)
        {
            Jzpm_list[i] = static_cast<double>(i) / static_cast<double>(Jzpm_list.size() - 1UL);
        }

        for(const auto& Jpmpm : Jpmpm_list)
        {       
            for(const auto& Jzpm : Jzpm_list)
            {
                auto startGS = std::chrono::steady_clock::now();

                std::cout << "Jpmpm: " << Jpmpm.real() << "; Jzpm: " << Jzpm.real() << std::endl;

                std::vector<QTensorNet::OpTerm> H_terms;

                for(const auto& [i, j, b] : latt)
                {                     
                    const auto& phi = phi_list[b];
                
                    const auto cos_phi = QTensorNet::complexType(std::cos(phi), 0.0);
                    const auto sin_phi = QTensorNet::complexType(std::sin(phi), 0.0);
                
                    H_terms.push_back({{i, J,  Sx_host}, {j, QTensorNet::complexType(1.0, 0.0), Sx_host}});
                    H_terms.push_back({{i, J,  Sy_host}, {j, QTensorNet::complexType(1.0, 0.0), Sy_host}});
                    H_terms.push_back({{i, dJ, Sz_host}, {j, QTensorNet::complexType(1.0, 0.0), Sz_host}});
                
                    H_terms.push_back({{i, QTensorNet::complexType(-2.0, 0.0) * Jpmpm * cos_phi, Sx_host}, {j, QTensorNet::complexType(1.0, 0.0), Sx_host}});
                    H_terms.push_back({{i, QTensorNet::complexType( 2.0, 0.0) * Jpmpm * cos_phi, Sy_host}, {j, QTensorNet::complexType(1.0, 0.0), Sy_host}});
                    H_terms.push_back({{i, QTensorNet::complexType( 2.0, 0.0) * Jpmpm * sin_phi, Sx_host}, {j, QTensorNet::complexType(1.0, 0.0), Sy_host}});
                    H_terms.push_back({{i, QTensorNet::complexType( 2.0, 0.0) * Jpmpm * sin_phi, Sy_host}, {j, QTensorNet::complexType(1.0, 0.0), Sx_host}});
                
                    H_terms.push_back({{i, QTensorNet::complexType(-1.0, 0.0) * Jzpm * cos_phi, Sx_host}, {j, QTensorNet::complexType(1.0, 0.0), Sz_host}});
                    H_terms.push_back({{i, QTensorNet::complexType(-1.0, 0.0) * Jzpm * cos_phi, Sz_host}, {j, QTensorNet::complexType(1.0, 0.0), Sx_host}});
                    H_terms.push_back({{i, QTensorNet::complexType(-1.0, 0.0) * Jzpm * sin_phi, Sy_host}, {j, QTensorNet::complexType(1.0, 0.0), Sz_host}});
                    H_terms.push_back({{i, QTensorNet::complexType(-1.0, 0.0) * Jzpm * sin_phi, Sz_host}, {j, QTensorNet::complexType(1.0, 0.0), Sy_host}});
                }
            
                QTensorNet::virtualModesGraphType H_graph(graph);
            
                std::vector<std::vector<QTensorNet::complexType>> H_tensors_host;
            
                QTensorNet::BuildOpTensors(H_graph, numSites, rootTTO, H_terms, H_tensors_host);

                std::vector<std::vector<int64_t>> physExtentsOp(numSites, std::vector<int64_t>{physExtent, physExtent});
            
                QTensorNet::TensorNetwork hamiltonian(physExtentsOp, H_graph, rootTTO, maxVirtualExtentTTO, 1UL, workSpaceLimitTTO);
            
                try
                {
                    hamiltonian.SetState(H_tensors_host);
                    hamiltonian.SetSVDConfig(absCutoffTTO, relCutoffTTO);
                    hamiltonian.Shrink();
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }
                
                H_terms.clear();
                H_tensors_host.clear();
                
                QTensorNet::TensorNetwork tts_ground(init_tts);

                try
                {                    
                    QTensorNet::complexType energy = hamiltonian.FindGroundStateUsingDMRG(&tts_ground, 
                                                                                          /*error_threshold*/ 1.0E-7, 
                                                                                          /*max_iter*/ 50,
                                                                                          /*max_virtual_extents*/ max_virtual_extents,
                                                                                          /*stream_num*/ 0UL,
                                                                                          /*cached*/ true,
                                                                                          /*verbose*/ 1UL, 
                                                                                          /*optimizerAttributes*/ optimizer_attributes,
                                                                                          /*numAutotuningIterations*/ 5);

                    std::cout << "The ground energy of the system: " << energy << std::endl;
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }

                auto finishGS = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsedGS = finishGS - startGS;
                std::cout << "Spent time for ground state evaluation: " << elapsedGS.count() << " s." << std::endl;

                auto startOBS = std::chrono::steady_clock::now();

                QTensorNet::CuTensorNetMethods::MPI_ = false;
                
                tts_ground.SetNumStreams(num_threads);

                DoTask(pool, thread_func_S_obs, tts_ground, num_threads, numSites);
                DoTask(pool, thread_func_SS_obs, tts_ground, num_threads, pairs.size());

                tts_ground.SetNumStreams(1UL);

                if(numProcs > 1)
                {
                    QTensorNet::CuTensorNetMethods::MPI_ = true;
                }

                std::cout << "<Psi_ground| S_i * S_j |Psi_ground>: " << std::endl;

                for(size_t k = 0UL; k < pairs.size(); ++k)
                {
                    auto [i, j] = pairs[k];
                    
                    std::cout << i << "\t" << j << "\t" << SS_obs[k] << std::endl;
                }

                std::cout << "<Psi_ground| S_i |Psi_ground>: " << std::endl;

                for(size_t k = 0UL; k < numSites; ++k)
                {
                    auto [Sx, Sy, Sz] = S_obs[k];
                    
                    std::cout << k << "\t" << Sx << "\t" << Sy << "\t" << Sz << std::endl;
                }

                auto finishOBS = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsedOBS = finishOBS - startOBS;
                std::cout << "Spent time for evaluation of observables: " << elapsedOBS.count() << " s.\n" << std::endl;
            }
        }

        HANDLE_CUDA_ERROR(cudaFree(Sx_device));
        HANDLE_CUDA_ERROR(cudaFree(Sy_device));
        HANDLE_CUDA_ERROR(cudaFree(Sz_device));
        HANDLE_CUDA_ERROR(cudaFree(SS_device));

        QTensorNet::CuTensorNetMethods::SendSignalToMPICommWorld(-1);

        auto finish = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "\nTotal spent time: " << elapsed.count() << " s." << std::endl;
    }
    else
    {
        QTensorNet::CuTensorNetMethods::MPIContractionHelper();
    }

    HANDLE_MPI_ERROR(MPI_Finalize());

    return 0;   
}