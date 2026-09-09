#include <fstream>
#include <chrono>
#include <functional>
#include <random>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"

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

        size_t numSites = 80UL;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtentTTS = 600L;
        int64_t maxVirtualExtentTTO = 200L;
        double absCutoffTTS = 0.0;
        double absCutoffTTO = 0.0;
        double relCutoffTTS = 1.0e-8;
        double relCutoffTTO = 1.0e-8;
        size_t numThreadsTTS = 1UL;
        size_t numThreadsTTO = 1UL;
        size_t workSpaceLimitTTS = 50UL * 1024UL;
        size_t workSpaceLimitTTO = 50UL * 1024UL;

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 1000},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 10000}};

        std::vector<size_t> max_virtual_extents = {20UL, 20UL, 
                                                   30UL, 30UL, 
                                                   50UL, 50UL, 
                                                   100UL, 100UL, 
                                                   200UL, 200UL, 
                                                   400UL, 400UL,
                                                   600UL, 600UL};
    
        size_t rootTTS = 76UL;
        size_t rootTTO = 76UL;

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

        std::vector<std::tuple<size_t, size_t, size_t>> latt = {{1, 3, 0}, {5, 7, 0}, {9, 11, 0}, {13, 15, 0}, {2, 16, 0}, {6, 18, 0}, {10, 20, 0}, {14, 22, 0}, 
                                                                {17, 25, 0}, {19, 27, 0}, {21, 29, 0}, {23, 31, 0}, {24, 32, 0}, {26, 34, 0}, {28, 36, 0}, 
                                                                {30, 38, 0}, {33, 41, 0}, {35, 43, 0}, {37, 45, 0}, {39, 47, 0}, {40, 48, 0}, {42, 50, 0}, 
                                                                {44, 52, 0}, {46, 54, 0}, {49, 57, 0}, {51, 59, 0}, {53, 61, 0}, {55, 63, 0}, {56, 64, 0}, 
                                                                {58, 66, 0}, {60, 68, 0}, {62, 70, 0}, {65, 73, 0}, {67, 75, 0}, {69, 77, 0}, {71, 79, 0}, 
                                                                {0, 72, 0}, {4, 74, 0}, {8, 76, 0}, {12, 78, 0}, {0, 2, 1}, {4, 6, 1}, {8, 10, 1}, {12, 14, 1}, 
                                                                {3, 17, 1}, {7, 19, 1}, {11, 21, 1}, {15, 23, 1}, {16, 24, 1}, {18, 26, 1}, {20, 28, 1}, 
                                                                {22, 30, 1}, {25, 33, 1}, {27, 35, 1}, {29, 37, 1}, {31, 39, 1}, {32, 40, 1}, {34, 42, 1}, 
                                                                {36, 44, 1}, {38, 46, 1}, {41, 49, 1}, {43, 51, 1}, {45, 53, 1}, {47, 55, 1}, {48, 56, 1}, 
                                                                {50, 58, 1}, {52, 60, 1}, {54, 62, 1}, {57, 65, 1}, {59, 67, 1}, {61, 69, 1}, {63, 71, 1}, 
                                                                {64, 72, 1}, {66, 74, 1}, {68, 76, 1}, {70, 78, 1}, {1, 73, 1}, {5, 75, 1}, {9, 77, 1}, 
                                                                {13, 79, 1}, {0, 1, 2}, {4, 5, 2}, {8, 9, 2}, {12, 13, 2}, {3, 6, 2}, {7, 10, 2}, {11, 14, 2}, 
                                                                {16, 17, 2}, {18, 19, 2}, {20, 21, 2}, {22, 23, 2}, {25, 26, 2}, {27, 28, 2}, {29, 30, 2}, 
                                                                {32, 33, 2}, {34, 35, 2}, {36, 37, 2}, {38, 39, 2}, {41, 42, 2}, {43, 44, 2}, {45, 46, 2}, 
                                                                {48, 49, 2}, {50, 51, 2}, {52, 53, 2}, {54, 55, 2}, {57, 58, 2}, {59, 60, 2}, {61, 62, 2}, 
                                                                {64, 65, 2}, {66, 67, 2}, {68, 69, 2}, {70, 71, 2}, {73, 74, 2}, {75, 76, 2}, {77, 78, 2}};

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

        QTensorNet::virtualModesGraphType graph = {{1, 3, 1}, {5, 7, 1}, {9, 11, 1}, {13, 15, 1}, {2, 16, 1}, {6, 18, 1}, 
                                                   {10, 20, 1}, {14, 22, 1}, {17, 25, 1}, {19, 27, 1}, {21, 29, 1}, {23, 31, 1}, 
                                                   {24, 32, 1}, {26, 34, 1}, {28, 36, 1}, {30, 38, 1}, {33, 41, 1}, {35, 43, 1}, 
                                                   {37, 45, 1}, {39, 47, 1}, {40, 48, 1}, {42, 50, 1}, {44, 52, 1}, {46, 54, 1}, 
                                                   {49, 57, 1}, {51, 59, 1}, {53, 61, 1}, {55, 63, 1}, {56, 64, 1}, {58, 66, 1}, 
                                                   {60, 68, 1}, {62, 70, 1}, {65, 73, 1}, {67, 75, 1}, {69, 77, 1}, {71, 79, 1}, 
                                                   {0, 2, 1}, {4, 6, 1}, {8, 10, 1}, {12, 14, 1}, {3, 17, 1}, {7, 19, 1}, 
                                                   {11, 21, 1}, {15, 23, 1}, {16, 24, 1}, {18, 26, 1}, {20, 28, 1}, {22, 30, 1}, 
                                                   {25, 33, 1}, {27, 35, 1}, {29, 37, 1}, {31, 39, 1}, {32, 40, 1}, {34, 42, 1}, 
                                                   {36, 44, 1}, {38, 46, 1}, {41, 49, 1}, {43, 51, 1}, {45, 53, 1}, {47, 55, 1}, 
                                                   {48, 56, 1}, {50, 58, 1}, {52, 60, 1}, {54, 62, 1}, {57, 65, 1}, {59, 67, 1}, 
                                                   {61, 69, 1}, {63, 71, 1}, {64, 72, 1}, {66, 74, 1}, {68, 76, 1}, {70, 78, 1}, 
                                                   {0, 1, 1}, {4, 5, 1}, {8, 9, 1}, {12, 13, 1}, {73, 74, 1}, {75, 76, 1}, {77, 78, 1}};

        std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

        QTensorNet::TensorNetwork init_tts(physExtentsVec, graph, rootTTS, maxVirtualExtentTTS, numThreadsTTS, workSpaceLimitTTS);

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
            
                QTensorNet::TensorNetwork hamiltonian(physExtentsOp, H_graph, rootTTO, maxVirtualExtentTTO, numThreadsTTO, workSpaceLimitTTO);
            
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
            
                H_tensors_host.clear();
                
                QTensorNet::TensorNetwork tts_ground(init_tts);

                try
                {                    
                    QTensorNet::complexType energy = hamiltonian.FindGroundStateUsingDMRG(&tts_ground, 
                                                                                          /*error_threshold*/ 1.0E-7, 
                                                                                          /*max_iter*/ 50,
                                                                                          /*max_virtual_extents*/ max_virtual_extents,
                                                                                          /*thread_num*/ 0UL,
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

                std::cout << "<Psi_ground| S_i * S_j |Psi_ground>: " << std::endl;

                for(size_t i = 0UL; i < numSites; ++i)
                {
                    for(size_t j = i + 1UL; j < numSites; ++j)
                    {
                        try
                        {
                            std::vector<int32_t> SiSjModes = {tts_ground.GetNode(i).physModes_[0],
                                                              tts_ground.GetNode(j).physModes_[0], 
                                                              tts_ground.GetNode(i).physModes_[0],
                                                              tts_ground.GetNode(j).physModes_[0]};
                            std::vector<int64_t> SiSjExtents = {2, 2, 2, 2};
                            
                            QTensorNet::complexType SS = tts_ground.ComputeMatrixElement(SS_device, 
                                                                                         SiSjModes, 
                                                                                         SiSjExtents,
                                                                                         nullptr,
                                                                                         /*thread_num*/ 0UL, 
                                                                                         /*optimizerAttributes*/ optimizer_attributes);

                            std::cout << i << "\t" << j << "\t" << SS << std::endl;
                        }
                        catch(const std::exception& ex)
                        {
                            std::cerr << ex.what() << std::endl;
                            std::exit(1);
                        }
                    }
                }

                std::cout << "<Psi_ground| S_i |Psi_ground>: " << std::endl;

                for(size_t i = 0UL; i < numSites; ++i)
                {
                    try
                    {
                        std::vector<int32_t> SiModes = {tts_ground.GetNode(i).physModes_[0], 
                                                        tts_ground.GetNode(i).physModes_[0]};
                        std::vector<int64_t> SiExtents = {2, 2};

                        QTensorNet::complexType Sx = tts_ground.ComputeMatrixElement(Sx_device, 
                                                                                     SiModes, 
                                                                                     SiExtents,
                                                                                     nullptr,
                                                                                     /*thread_num*/ 0UL, 
                                                                                     /*optimizerAttributes*/ optimizer_attributes);

                        QTensorNet::complexType Sy = tts_ground.ComputeMatrixElement(Sy_device, 
                                                                                     SiModes, 
                                                                                     SiExtents,
                                                                                     nullptr,
                                                                                     /*thread_num*/ 0UL, 
                                                                                     /*optimizerAttributes*/ optimizer_attributes);
                        
                        QTensorNet::complexType Sz = tts_ground.ComputeMatrixElement(Sz_device, 
                                                                                     SiModes, 
                                                                                     SiExtents,
                                                                                     nullptr,
                                                                                     /*thread_num*/ 0UL, 
                                                                                     /*optimizerAttributes*/ optimizer_attributes);

                        std::cout << i << "\t" << Sx << "\t" << Sy << "\t" << Sz << std::endl;
                    }
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        std::exit(1);
                    }
                }

                std::cout << std::endl;
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