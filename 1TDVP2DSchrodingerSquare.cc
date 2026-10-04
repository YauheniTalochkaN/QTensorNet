#include <fstream>
#include <chrono>
#include <functional>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"
#include "TaylorSeriesIntegrator.hh"

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
        
        size_t Nx = 8UL;
        size_t Ny = 8UL;
        size_t numSites = Nx * Ny;
        size_t numSitesExt = 94;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtentVec = 150L;
        int64_t maxVirtualExtentOp = 100L;
        double absCutoffVec = 0.0;
        double absCutoffOp = 0.0;
        double relCutoffVec = 1.0e-8;
        double relCutoffOp = 1.0e-8;
        size_t workSpaceLimitVec = 50UL * 1024UL;
        size_t workSpaceLimitOp = 50UL * 1024UL;

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 10},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 1000}};
        
        size_t num_iter = 100UL;
        double tmax = 1.0;
        double hz = 3.04438;

        double dt = tmax / static_cast<double>(num_iter);

        auto latt = SquareLattice(Nx, Ny);

        std::vector<std::vector<int64_t>> physExtentsVec(numSitesExt);

        for(size_t i = 0UL; i < numSites; ++i)
        {
            physExtentsVec[i].push_back(physExtent);
        }

        for(size_t i = numSites; i < numSitesExt; ++i)
        {
            physExtentsVec[i].push_back(1L);
        }

        size_t root = 0UL;

        QTensorNet::virtualModesGraphType graph = {{64, 0, 2}, {64, 8, 2}, {64, 1, 2}, {64, 9, 2}, {65, 16, 2}, {65, 24, 2}, 
                                                   {65, 17, 2}, {65, 25, 2}, {66, 32, 2}, {66, 40, 2}, {66, 33, 2}, {66, 41, 2}, 
                                                   {67, 48, 2}, {67, 56, 2}, {67, 49, 2}, {67, 57, 2}, {68, 2, 2}, {68, 10, 2}, 
                                                   {68, 3, 2}, {68, 11, 2}, {69, 18, 2}, {69, 26, 2}, {69, 19, 2}, {69, 27, 2}, 
                                                   {70, 34, 2}, {70, 42, 2}, {70, 35, 2}, {70, 43, 2}, {71, 50, 2}, {71, 58, 2}, 
                                                   {71, 51, 2}, {71, 59, 2}, {72, 4, 2}, {72, 12, 2}, {72, 5, 2}, {72, 13, 2}, 
                                                   {73, 20, 2}, {73, 28, 2}, {73, 21, 2}, {73, 29, 2}, {74, 36, 2}, {74, 44, 2}, 
                                                   {74, 37, 2}, {74, 45, 2}, {75, 52, 2}, {75, 60, 2}, {75, 53, 2}, {75, 61, 2}, 
                                                   {76, 6, 2}, {76, 14, 2}, {76, 7, 2}, {76, 15, 2}, {77, 22, 2}, {77, 30, 2}, 
                                                   {77, 23, 2}, {77, 31, 2}, {78, 38, 2}, {78, 46, 2}, {78, 39, 2}, {78, 47, 2}, 
                                                   {79, 54, 2}, {79, 62, 2}, {79, 55, 2}, {79, 63, 2}, {80, 64, 10}, {80, 68, 10}, 
                                                   {65, 81, 10}, {69, 81, 10}, {80, 82, 10}, {81, 82, 10}, {72, 83, 10}, {76, 83, 10}, 
                                                   {73, 84, 10}, {77, 84, 10}, {83, 85, 10}, {84, 85, 10}, {85, 86, 10}, {82, 86, 10}, 
                                                   {66, 87, 10}, {70, 87, 10}, {67, 88, 10}, {71, 88, 10}, {87, 89, 10}, {88, 89, 10}, 
                                                   {74, 90, 10}, {78, 90, 10}, {75, 91, 10}, {79, 91, 10}, {90, 92, 10}, {91, 92, 10}, 
                                                   {89, 93, 10}, {92, 93, 10}, {86, 93, 10}};


        QTensorNet::TensorNetwork psi(physExtentsVec, graph, root, maxVirtualExtentVec, 1UL, workSpaceLimitVec);

        std::vector<std::vector<QTensorNet::complexType>> psi_tensors_host;

        std::cout << "\nInitializing Psi tensors..." << std::endl;

        for(size_t i = 0UL; i < numSitesExt; ++i)
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
    
        QTensorNet::BuildOpTensors(H_graph, numSitesExt, root, H_terms, H_tensors_host);
        
        std::vector<std::vector<int64_t>> physExtentsOp(numSitesExt);

        for(size_t i = 0UL; i < numSites; ++i)
        {
            physExtentsOp[i].push_back(physExtent);
            physExtentsOp[i].push_back(physExtent);
        }

        for(size_t i = numSites; i < numSitesExt; ++i)
        {
            physExtentsOp[i].push_back(1L);
            physExtentsOp[i].push_back(1L);
        }
    
        QTensorNet::TensorNetwork hamiltonian(physExtentsOp, H_graph, root, maxVirtualExtentOp, 1UL, workSpaceLimitOp);
    
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
        
        H_terms.clear();
        H_tensors_host.clear();

        auto finishH = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedH = finishH - startH;
        std::cout << "\nTotal spent time for Hamiltonian components building: " << elapsedH.count() << " s." << std::endl;

        //---lambda functions for observables----------------------------------------------

        std::vector<std::vector<double>> expectations(numSites);

        auto eval_expectation_sigmaZ_psi = [&psi, SigmaZ_device, &expectations, &optimizer_attributes, &numSites] 
        {
            for(size_t j = 0UL; j < numSites; ++j)
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
                                                                0UL,
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

        std::ofstream outFile("ExpectedSigmaZ_2DChain.txt");

        if(!outFile.is_open()) 
        {
            std::cerr << "Error when creating expectations file." << std::endl;

            return 1;
        }

        outFile << std::fixed << std::setprecision(10);
        
        eval_expectation_sigmaZ_psi();

        outFile << 0.0 << "\t";

        for(size_t j = 0UL; j < numSites; ++j) 
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
                psi.UpdateUsingTDVP(&hamiltonian, solver, dt, root, 4UL, true, false, 0UL, optimizer_attributes, 5);

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

            eval_expectation_sigmaZ_psi();

            outFile << static_cast<double>(iter + 1UL) * dt << "\t";

            for(size_t j = 0UL; j < numSites; ++j) 
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