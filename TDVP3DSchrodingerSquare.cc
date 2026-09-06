#include <fstream>
#include <chrono>
#include <functional>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"
#include "TaylorSeriesIntegrator.hh"

void DoTask(const std::function<void(size_t, size_t, size_t)>& task, size_t num_threads, size_t num_sites)
{
    std::vector<std::thread> threads;
    
    size_t chunk_size = (num_sites + num_threads - 1) / num_threads;

    for(size_t th = 0; th < num_threads; ++th) 
    {
        size_t start = th * chunk_size;
        size_t end = std::min(start + chunk_size, num_sites);

        if (start < num_sites) 
        {
            threads.emplace_back(task, th, start, end);
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
}

std::vector<std::tuple<size_t, size_t, size_t>> SquareLattice(int64_t Nx, int64_t Ny)
{
    size_t Nbond = 2 * Nx * Ny;
    
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
            latt.emplace_back(i + j * Nx, mod(i + 1, Nx) + j * Nx, 0);
            latt.emplace_back(i + j * Nx, i + mod(j + 1, Ny) * Nx, 1);
        } 
    }
    
    if(latt.size() != Nbond) std::cerr << "SquareLattice: Wrong number of bonds." << std::endl;
    
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
        
        size_t numSites = 64UL;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtentVec = 600L;
        int64_t maxVirtualExtentOp = 100L;
        double absCutoffVec = 0.0;
        double absCutoffOp = 0.0;
        double relCutoffVec = 1.0e-8;
        double relCutoffOp = 1.0e-8;
        size_t numThreadsVec = 1UL;
        size_t numThreadsOp = 1UL;
        size_t workSpaceLimitVec = 50UL * 1024UL;
        size_t workSpaceLimitOp = 50UL * 1024UL;

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 10},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 1000}};
        
        size_t num_iter = 100UL;
        double tmax = 1.0;
        double hz = 3.04438;

        double dt = tmax / static_cast<double>(num_iter);

        auto latt = SquareLattice(8L, 8L);

        std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

        QTensorNet::virtualModesGraphType graph;

        size_t root = numSites / 2UL;

        for(size_t i = 0UL; i < numSites - 1UL; ++i)
        {
            graph.insert(std::make_tuple(i, i + 1UL, 1L));
        }

        QTensorNet::TensorNetwork psi(physExtentsVec, graph, root, maxVirtualExtentVec, numThreadsVec, workSpaceLimitVec);

        std::vector<std::vector<QTensorNet::complexType>> psi_tensors_host;

        std::cout << "\nInitializing Psi tensors..." << std::endl;

        for(size_t i = 0UL; i < numSites; ++i)
        {        
            std::vector<QTensorNet::complexType> data_host(psi.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

            data_host[0] = QTensorNet::complexType(1.0, 0.0);

            psi_tensors_host.push_back(data_host);

            std::cout << "Site " << i << ": tensor[0] = (" << psi_tensors_host[i][0].real() 
                      << ", " << psi_tensors_host[i][0].imag() << "), tensor[1] = (" 
                      << psi_tensors_host[i][1].real() << ", " << psi_tensors_host[i][1].imag() << ")" << std::endl;
        }

        try
        {
            psi.SetState(psi_tensors_host);
            psi.SetSVDConfig(absCutoffVec, relCutoffVec);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }

        //-----------------------------------------------------------------------------------

        auto SigmaZ_host = QTensorNet::BasisGates::SigmaZ();
        void* SigmaZ_device = QTensorNet::CuArrayMethods::VectorToGPUArray(SigmaZ_host);

        auto SigmaX_host = QTensorNet::BasisGates::SigmaX();

        //---Hamiltonian-------------------------------------------------------------------

        std::cout << "\nBuilding Hamiltonian components..." << std::endl;

        auto startH = std::chrono::steady_clock::now();

        std::vector<QTensorNet::OpTerm> H_terms;

        for(size_t i = 0UL; i < numSites; ++i)
        {                     
            H_terms.push_back({{i, QTensorNet::complexType(-hz, 0.0), SigmaZ_host}});
        }
    
        for(const auto& [i, j, b] : latt)
        {                                      
            H_terms.push_back({{i, QTensorNet::complexType(-1.0, 0.0), SigmaX_host}, {j, QTensorNet::complexType(1.0, 0.0), SigmaX_host}});
        }
        
        QTensorNet::virtualModesGraphType H_graph(graph);
    
        std::vector<std::vector<QTensorNet::complexType>> H_tensors_host;
    
        QTensorNet::BuildOpTensors(H_graph, numSites, root, H_terms, H_tensors_host);
        
        std::vector<std::vector<int64_t>> physExtentsOp(numSites, std::vector<int64_t>{physExtent, physExtent});
    
        QTensorNet::TensorNetwork hamiltonian(physExtentsOp, H_graph, root, maxVirtualExtentOp, numThreadsOp, workSpaceLimitOp);
    
        try
        {
            hamiltonian.SetState(H_tensors_host);
            hamiltonian.SetSVDConfig(absCutoffOp, relCutoffOp);
            hamiltonian.Shrink();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }
    
        H_tensors_host.clear();

        hamiltonian.Save("./hamiltonian");

        //QTensorNet::TensorNetwork hamiltonian("./hamiltonian");

        auto finishH = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedH = finishH - startH;
        std::cout << "\nTotal spent time for Hamiltonian components building: " << elapsedH.count() << " s." << std::endl;

        //---lambda functions for observables----------------------------------------------

        std::vector<std::vector<double>> expectations(numSites);

        auto thread_func_expectation_sigmaZ_psi = [&psi, SigmaZ_device, &expectations, &optimizer_attributes](size_t thread_num, size_t start, size_t end) 
        {
            for(size_t j = start; j < end; ++j)
            {           
                QTensorNet::complexType expectationValue(0.0, 0.0);

                try
                {
                    std::vector<int32_t> SzModes = {psi.GetNode(j).physModes_[0], 
                                                    psi.GetNode(j).physModes_[0]};
                    std::vector<int64_t> SzExtents = {2, 2};
                    
                    expectationValue = psi.ComputeMatrixElement(SigmaZ_device, 
                                                                SzModes,  
                                                                SzExtents,
                                                                nullptr,
                                                                thread_num,
                                                                optimizer_attributes);
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }

                expectations[j].push_back(expectationValue.real());
            }
        };

        //---------------------------------------------------------------------------------

        std::cout << "\nCalculating the energy of the spin net at the initial Psi state vector..." << std::endl;

        try
        {             
            QTensorNet::complexType energy = psi.ComputeMatrixElement(&hamiltonian, 
                                                                      nullptr, 
                                                                      0UL, 
                                                                      optimizer_attributes);

            std::cout << "Energy of the system: " << energy << std::endl;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }

        //---------------------------------------------------------------------------------

        std::ofstream outFile("ExpectedSigmaZ_3DChain.txt");

        if (!outFile.is_open()) 
        {
            std::cerr << "Error when creating expectations file." << std::endl;

            return 1;
        }

        outFile << std::fixed << std::setprecision(10);
        
        DoTask(thread_func_expectation_sigmaZ_psi, numThreadsVec, numSites);

        outFile << 0.0 << "\t";

        for (size_t j = 0UL; j < numSites; ++j) 
        {
            if(j < numSites-1) outFile << expectations[j][0] << "\t";
            else outFile << expectations[j][0] << std::endl;
        }

        //---------------------------------------------------------------------------------

        std::cout << "\nApplying unitary evolution operator and obtaining physical entities..." << std::endl; 
        
        QTensorNet::Integrators::TaylorSeriesIntegrator solver(1.0E-8);

        for(size_t iter = 0UL; iter < num_iter; ++iter)
        {
            std::cout << "Iteration: " << iter + 1UL << "/" << num_iter << "\r" << std::flush;

            try
            {
                psi.UpdateUsingTDVP(&hamiltonian, solver, dt, 0UL, 2UL, true, false, 0UL, optimizer_attributes, 5);

                auto [norm_device, descNorm] = psi.GetDensityMatrix({}, true, 0UL, optimizer_attributes);
                auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

                HANDLE_CUDA_ERROR(cudaFree(norm_device));

                psi *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }

            DoTask(thread_func_expectation_sigmaZ_psi, numThreadsVec, numSites);

            outFile << static_cast<double>(iter + 1UL) * dt << "\t";

            for (size_t j = 0UL; j < numSites; ++j) 
            {
                if(j < numSites-1) outFile << expectations[j][iter + 1UL] << "\t";
                else outFile << expectations[j][iter + 1UL] << std::endl;
            }
        }

        outFile.close();

        std::cout << "\n\nCalculating the energy of the spin net at the final Psi state vector..." << std::endl;

        try
        {            
            QTensorNet::complexType energy = psi.ComputeMatrixElement(&hamiltonian);

            std::cout << "Energy of the system: " << energy << std::endl;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }

        HANDLE_CUDA_ERROR(cudaFree(SigmaZ_device));

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