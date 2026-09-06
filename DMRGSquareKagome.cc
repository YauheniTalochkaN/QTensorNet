#include <fstream>
#include <chrono>
#include <functional>
#include <random>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"

std::vector<std::tuple<size_t, size_t, size_t>> SquareKagomeLattice(int64_t Nx, int64_t Ny)
{
    size_t Nbond = 22 * Nx * Ny;
    
    std::vector<std::tuple<size_t, size_t, size_t>> latt;
    latt.reserve(Nbond);

    auto mod = [](int64_t i, int64_t N)
    {
        return (i < 0) ? (i % N + N) % N : i % N;
    };
    
    for(int64_t i = 0; i < Nx; ++i)
    {
        for(int64_t j = 0; j < Ny; ++j)
        {
            latt.emplace_back(1 + 7 * (i + j * Nx), 4 + 7 * (i + j * Nx), 0);
            latt.emplace_back(2 + 7 * (i + j * Nx), 5 + 7 * (i + j * Nx), 0);
            latt.emplace_back(4 + 7 * (i + j * Nx), 3 + 7 * (mod(i + 1, Nx) + j * Nx), 0);
            latt.emplace_back(5 + 7 * (i + j * Nx), 0 + 7 * (i + mod(j + 1, Ny) * Nx), 0);
                        
            latt.emplace_back(0 + 7 * (i + j * Nx), 1 + 7 * (i + j * Nx), 1);
            latt.emplace_back(1 + 7 * (i + j * Nx), 2 + 7 * (i + j * Nx), 1);
            latt.emplace_back(2 + 7 * (i + j * Nx), 3 + 7 * (i + j * Nx), 1);
            latt.emplace_back(3 + 7 * (i + j * Nx), 0 + 7 * (i + j * Nx), 1);
                        
            latt.emplace_back(2 + 7 * (i + j * Nx), 4 + 7 * (i + j * Nx), 2);
            latt.emplace_back(3 + 7 * (i + j * Nx), 5 + 7 * (i + j * Nx), 2);
            latt.emplace_back(4 + 7 * (i + j * Nx), 0 + 7 * (mod(i + 1, Nx) + j * Nx), 2);
            latt.emplace_back(5 + 7 * (i + j * Nx), 1 + 7 * (i + mod(j + 1, Ny) * Nx), 2);
                        
            latt.emplace_back(0 + 7 * (i + j * Nx), 2 + 7 * (i + j * Nx), 3);
            latt.emplace_back(1 + 7 * (i + j * Nx), 3 + 7 * (i + j * Nx), 3);
                        
            latt.emplace_back(0 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 4);
            latt.emplace_back(1 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 4);
            latt.emplace_back(2 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 4);
            latt.emplace_back(3 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 4);
                        
            latt.emplace_back(4 + 7 * (i + j * Nx), 5 + 7 * (i + j * Nx), 5);
            latt.emplace_back(4 + 7 * (i + j * Nx), 5 + 7 * (mod(i + 1, Nx) + j * Nx), 5);
            latt.emplace_back(5 + 7 * (i + j * Nx), 4 + 7 * (i + mod(j + 1, Ny) * Nx), 5);
            latt.emplace_back(5 + 7 * (i + j * Nx), 4 + 7 * (mod(i - 1, Nx) + mod(j + 1, Ny) * Nx), 5);
        } 
    }
    
    if(latt.size() != Nbond) std::cerr << "SquareKagomeLattice: Wrong number of bonds." << std::endl;
    
    return latt;
}

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

        size_t numSites = 112UL;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtentMPS = 1500L;
        int64_t maxVirtualExtentMPO = 200L;
        double absCutoffMPS = 0.0;
        double absCutoffMPO = 0.0;
        double relCutoffMPS = 1.0e-8;
        double relCutoffMPO = 1.0e-8;
        size_t numThreadsMPS = 1UL;
        size_t numThreadsMPO = 1UL;
        size_t workSpaceLimitMPS = 50UL * 1024UL;
        size_t workSpaceLimitMPO = 50UL * 1024UL;

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 1000},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 10000}};

        std::vector<size_t> max_virtual_extents = {20UL, 20UL, 
                                                   30UL, 30UL, 
                                                   50UL, 50UL, 
                                                   100UL, 100UL, 
                                                   300UL, 300UL, 
                                                   600UL, 600UL, 
                                                   1000UL, 1000UL,
                                                   1500UL, 1500UL};

        std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

        std::vector<double> Jlist = {0.012, 0.694, 0.971, 1.000, 0.894, 0.182};
        
        double Junit = 170.0;

        for(auto& it : Jlist)
        {
            it *= Junit;
        }

        double gmuB = 2.0 / 0.086 * 0.05788; 

        size_t rootMPS = numSites / 2UL;
        size_t rootMPO = numSites / 2UL;

        auto latt = SquareKagomeLattice(4L, 4L);

        QTensorNet::virtualModesGraphType graph;

        for(size_t i = 0UL; i < numSites - 1UL; ++i)
        {
            graph.insert(std::make_tuple(i, i + 1UL, 1L));
        }

        QTensorNet::TensorNetwork init_mps(physExtentsVec, graph, rootMPS, maxVirtualExtentMPS, numThreadsMPS, workSpaceLimitMPS);

        std::vector<std::vector<QTensorNet::complexType>> mps_tensors_host;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        std::cout << "\nInitializing MPS tensors for initial state..." << std::endl;

        for (size_t i = 0UL; i < numSites; ++i)
        {        
            std::vector<QTensorNet::complexType> data_host(init_mps.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

            data_host[0] = QTensorNet::complexType(dist(gen), dist(gen));
            data_host[1] = QTensorNet::complexType(dist(gen), dist(gen));

            mps_tensors_host.push_back(data_host);

            std::cout << "Site " << i << ": tensor[0] = (" << mps_tensors_host[i][0].real() 
                      << ", " << mps_tensors_host[i][0].imag() << "), tensor[1] = (" 
                      << mps_tensors_host[i][1].real() << ", " << mps_tensors_host[i][1].imag() << ")" << std::endl;
        }

        try
        {
            init_mps.SetState(mps_tensors_host);
            init_mps.SetSVDConfig(absCutoffMPS, relCutoffMPS);

            auto [norm_device, descNorm] = init_mps.GetDensityMatrix({}, true, 0UL, optimizer_attributes);
            auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

            HANDLE_CUDA_ERROR(cudaFree(norm_device));

            init_mps *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);
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

        std::cout << "\nLooking for the ground MPS vector of the hamiltonian..." << std::endl;

        double B = 0.0;
        double dB = 5.0;

        while(B <= 500.0)
        {       
            auto startGS = std::chrono::steady_clock::now();
            
            std::cout << "The magnetic induction: " << B << std::endl;

            std::vector<QTensorNet::OpTerm> H_terms;

            for(size_t i = 0UL; i < numSites; ++i)
            {                     
                H_terms.push_back({{i, QTensorNet::complexType(-gmuB * B, 0.0), Sz_host}});
            }
                
            for(const auto& [i, j, b] : latt)
            {                                    
                const auto J = QTensorNet::complexType(Jlist[b], 0.0);
                
                H_terms.push_back({{i, J, Sx_host}, {j, QTensorNet::complexType(1.0, 0.0), Sx_host}});
                H_terms.push_back({{i, J, Sy_host}, {j, QTensorNet::complexType(1.0, 0.0), Sy_host}});
                H_terms.push_back({{i, J, Sz_host}, {j, QTensorNet::complexType(1.0, 0.0), Sz_host}});
            }
        
            QTensorNet::virtualModesGraphType H_graph(graph);
        
            std::vector<std::vector<QTensorNet::complexType>> H_tensors_host;
        
            QTensorNet::BuildOpTensors(H_graph, numSites, rootMPO, H_terms, H_tensors_host);
            
            std::vector<std::vector<int64_t>> physExtentsOp(numSites, std::vector<int64_t>{physExtent, physExtent});
        
            QTensorNet::TensorNetwork hamiltonian_mpo(physExtentsOp, H_graph, rootMPO, maxVirtualExtentMPO, numThreadsMPO, workSpaceLimitMPO);
        
            try
            {
                hamiltonian_mpo.SetState(H_tensors_host);
                hamiltonian_mpo.SetSVDConfig(absCutoffMPO, relCutoffMPO);
                hamiltonian_mpo.Shrink();
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }
        
            H_tensors_host.clear();

            QTensorNet::TensorNetwork mps_ground(init_mps);

            try
            {                    
                QTensorNet::complexType energy = hamiltonian_mpo.FindGroundStateUsingDMRG(&mps_ground, 
                                                                                          /*error_threshold*/ 1.0E-5, 
                                                                                          /*max_iter*/ 20,
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
                        std::vector<int32_t> SiSjModes = {mps_ground.GetNode(i).physModes_[0],
                                                          mps_ground.GetNode(j).physModes_[0], 
                                                          mps_ground.GetNode(i).physModes_[0],
                                                          mps_ground.GetNode(j).physModes_[0]};
                        std::vector<int64_t> SiSjExtents = {2, 2, 2, 2};
                        
                        QTensorNet::complexType SS = mps_ground.ComputeMatrixElement(SS_device, 
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
                    std::vector<int32_t> SiModes = {mps_ground.GetNode(i).physModes_[0], 
                                                    mps_ground.GetNode(i).physModes_[0]};
                    std::vector<int64_t> SiExtents = {2, 2};

                    QTensorNet::complexType Sx = mps_ground.ComputeMatrixElement(Sx_device, 
                                                                                 SiModes, 
                                                                                 SiExtents,
                                                                                 nullptr,
                                                                                 /*thread_num*/ 0UL, 
                                                                                 /*optimizerAttributes*/ optimizer_attributes);

                    QTensorNet::complexType Sy = mps_ground.ComputeMatrixElement(Sy_device, 
                                                                                 SiModes, 
                                                                                 SiExtents,
                                                                                 nullptr,
                                                                                 /*thread_num*/ 0UL, 
                                                                                 /*optimizerAttributes*/ optimizer_attributes);
                    
                    QTensorNet::complexType Sz = mps_ground.ComputeMatrixElement(Sz_device, 
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

            B += dB;
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