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
        
        size_t numSites = 42UL;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtentMat = 300L;
        int64_t maxVirtualExtentOp = 200L;
        double absCutoffMat = 0.0;
        double absCutoffOp = 0.0;
        double relCutoffMat = 1.0e-8;
        double relCutoffOp = 1.0e-8;
        size_t numThreadsMat = 1UL;
        size_t numThreadsOp = 1UL;
        size_t workSpaceLimitMat = 50UL * 1024UL;
        size_t workSpaceLimitOp = 50UL * 1024UL;

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 10},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 1000}};
        
        size_t num_iter = 2000UL;
        double tmax = 100.0;
        double J = 1.0;

        double dt = tmax / static_cast<double>(num_iter);

        std::vector<std::tuple<double, double, double>> positions = {{ 0.816497,  0.816497, 6.53197}, {-1.115360,  0.298858, 6.53197}, 
                                                                     { 0.298858, -1.115360, 6.53197}, {-0.298858,  1.115360, 8.16497}, 
                                                                     { 1.115360, -0.298858, 8.16497}, {-0.816497, -0.816497, 4.49073}, 
                                                                     {-1.414210,  1.414210, 6.12372}, { 0.000000,  0.000000, 6.12372}, 
                                                                     {-0.597717,  2.230710, 7.75672}, { 1.414210, -1.414210, 6.12372}, 
                                                                     { 0.816497,  0.816497, 7.75672}, { 2.230710, -0.597717, 7.75672}, 
                                                                     { 1.632990,  1.632990, 9.38971}, {-1.931850, -0.517638, 6.12372}, 
                                                                     {-2.529570,  1.713070, 7.75672}, {-0.517638, -1.931850, 6.12372}, 
                                                                     {-1.115360,  0.298858, 7.75672}, { 0.298858, -1.115360, 7.75672}, 
                                                                     {-0.298858,  1.115360, 9.38971}, { 1.713070, -2.529570, 7.75672}, 
                                                                     { 1.115360, -0.298858, 9.38971}, {-1.931850, -0.517638, 4.89898}, 
                                                                     {-2.529570,  1.713070, 6.53197}, {-0.517638, -1.931850, 4.89898}, 
                                                                     {-1.713070,  2.529570, 8.16497}, { 1.713070, -2.529570, 6.53197}, 
                                                                     { 0.517638,  1.931850, 9.79796}, { 2.529570, -1.713070, 8.16497}, 
                                                                     { 1.931850,  0.517638, 9.79796}, {-2.449490, -2.449490, 4.89898}, 
                                                                     {-3.047210, -0.218780, 6.53197}, {-3.644920,  2.011930, 8.16497}, 
                                                                     {-1.632990, -1.632990, 6.53197}, {-2.230710,  0.597717, 8.16497}, 
                                                                     {-0.218780, -3.047210, 6.53197}, {-0.816497, -0.816497, 8.16497}, 
                                                                     {-1.414210,  1.414210, 9.79796}, { 0.597717, -2.230710, 8.16497}, 
                                                                     { 0.000000,  0.000000, 9.79796}, { 2.011930, -3.644920, 8.16497}, 
                                                                     { 1.414210, -1.414210, 9.79796}, { 0.816497,  0.816497, 11.4310}};

        QTensorNet::virtualModesGraphType graph;

        size_t root = numSites / 2UL;

        for(size_t i = 0UL; i < numSites - 1UL; ++i)
        {
            graph.insert(std::make_tuple(i, i + 1UL, 1L));
        }

        std::vector<std::vector<int64_t>> physExtentsMat(numSites, std::vector<int64_t>{physExtent, physExtent});

        QTensorNet::TensorNetwork rho(physExtentsMat, graph, root, maxVirtualExtentMat, numThreadsMat, workSpaceLimitMat);

        std::vector<std::vector<QTensorNet::complexType>> rho_tensors_host;

        std::cout << "\nInitializing Rho tensors..." << std::endl;

        for(size_t i = 0UL; i < numSites; ++i)
        {        
            std::vector<QTensorNet::complexType> data_host(rho.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

            if(i == 0UL) 
            {
                data_host[3] = QTensorNet::complexType(1.0, 0.0);
            }
            else
            {
                data_host[0] = QTensorNet::complexType(0.5, 0.0);
                data_host[3] = QTensorNet::complexType(0.5, 0.0);
            }

            rho_tensors_host.push_back(data_host);

            std::cout << "Site " << i << ": ";

            for(size_t j = 0UL; j < rho_tensors_host[i].size(); ++j)
            {
                std::cout << "tensor[" << std::to_string(j) << "] = (" << rho_tensors_host[i][j].real() << ", " << rho_tensors_host[i][j].imag() << ")";
                
                if(j + 1UL < rho_tensors_host[i].size())
                {
                    std::cout << "; ";
                }
            }

            std::cout << std::endl;
        }

        try
        {
            rho.SetState(rho_tensors_host);
            rho.SetSVDConfig(absCutoffMat, relCutoffMat);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        } 

        //QTensorNet::TensorNetwork rho("./rho");

        std::vector<QTensorNet::complexType> rhoC0_host = {QTensorNet::complexType(0.0, 0.0),
                                                           QTensorNet::complexType(0.0, 0.0),
                                                           QTensorNet::complexType(0.0, 0.0),
                                                           QTensorNet::complexType(1.0, 0.0)};

        void* rhoC0_device = QTensorNet::CuArrayMethods::VectorToGPUArray(rhoC0_host);

        //---Unit operator-------------------------------------------------------------------

        std::vector<std::vector<int64_t>> physExtentsOp(numSites, std::vector<int64_t>{physExtent, physExtent});

        QTensorNet::TensorNetwork unitOp(physExtentsOp, graph, root, maxVirtualExtentOp, numThreadsOp, workSpaceLimitOp);

        std::vector<std::vector<QTensorNet::complexType>> unitOp_tensors_host;

        for (size_t i = 0UL; i < numSites; ++i)
        {
            std::vector<QTensorNet::complexType> data_host(unitOp.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

            data_host[0] = QTensorNet::complexType(1.0, 0.0);
            data_host[3] = QTensorNet::complexType(1.0, 0.0);

            unitOp_tensors_host.push_back(data_host);
        }

        try
        {
            unitOp.SetState(unitOp_tensors_host);
            unitOp.SetSVDConfig(absCutoffOp, relCutoffOp);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }

        //-----------------------------------------------------------------------------------
       
        auto Sz_host = QTensorNet::BasisGates::SigmaZ(0.5);
        void* Sz_device = QTensorNet::CuArrayMethods::VectorToGPUArray(Sz_host);

        auto Sminus_host = QTensorNet::BasisGates::SigmaMinus(0.5);
        auto Splus_host = QTensorNet::BasisGates::SigmaPlus(0.5);

        auto build_liouvillian = [&unitOp]
                                 (const QTensorNet::TensorNetwork& hamiltonian, QTensorNet::TensorNetwork** Liouvillian)
                                 {
                                      if(*Liouvillian != nullptr)
                                      {
                                          delete *Liouvillian;
                                      }
                                     
                                      *Liouvillian = unitOp.EvaluateTensorNetProduct(&hamiltonian);
                                     
                                      QTensorNet::TensorNetwork* Liouvillian_right = hamiltonian.EvaluateTensorNetProduct(&unitOp, {true, false});
                                      *Liouvillian_right *= QTensorNet::complexType(-1.0, 0.0);
                                      
                                      **Liouvillian += *Liouvillian_right;
                                      
                                      delete Liouvillian_right;
                                 };

        //---Hamiltonian-------------------------------------------------------------------

        std::cout << "\nBuilding Hamiltonian components..." << std::endl;

        auto startH = std::chrono::steady_clock::now();

        std::vector<QTensorNet::OpTerm> H_terms;

        for(size_t i = 0UL; i < numSites; ++i)
        {
            for(size_t j = i + 1UL; j < numSites; ++j)
            {           
                auto [xi, yi, zi] = positions.at(i);
                auto [xj, yj, zj] = positions.at(j);

                double dx = xj - xi;
                double dy = yj - yi;
                double dz = zj - zi;

                double dr3 = std::pow(dx * dx + dy * dy + dz * dz, 1.5);

                double dr = std::pow(dx * dx + dy * dy + dz * dz, 0.5);

                double cos_tetha = dz / dr;

                double Jzz = J * (1.0 - 3.0 * cos_tetha * cos_tetha) / dr3;
                
                H_terms.push_back({{i, QTensorNet::complexType(Jzz, 0.0), Sz_host}, {j, QTensorNet::complexType(1.0, 0.0), Sz_host}});
                H_terms.push_back({{i, QTensorNet::complexType(-Jzz / 4.0, 0.0), Splus_host}, {j, QTensorNet::complexType(1.0, 0.0), Sminus_host}});
                H_terms.push_back({{i, QTensorNet::complexType(-Jzz / 4.0, 0.0), Sminus_host}, {j, QTensorNet::complexType(1.0, 0.0), Splus_host}});
            }
        }
        
        QTensorNet::virtualModesGraphType H_graph(graph);
    
        std::vector<std::vector<QTensorNet::complexType>> H_tensors_host;
    
        QTensorNet::BuildOpTensors(H_graph, numSites, root, H_terms, H_tensors_host);
    
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

        auto thread_func_expectation_sigmaZ_rho = [&rho, Sz_device, &expectations, &optimizer_attributes](size_t thread_num, size_t start, size_t end) 
        {
            for(size_t j = start; j < end; ++j)
            {           
                QTensorNet::complexType expectationValue(0.0, 0.0);

                try
                {
                    std::vector<int32_t> SzModes = {rho.GetNode(j).physModes_[1], 
                                                    rho.GetNode(j).physModes_[1]};
                    std::vector<int64_t> SzExtents = {2, 2};
                    
                    expectationValue = QTensorNet::complexType(2.0, 0.0) * rho.ComputeOperatorTrace(Sz_device, 
                                                                                                    SzModes,  
                                                                                                    SzExtents,
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
        
        QTensorNet::TensorNetwork* Liouvillian = nullptr;

        std::cout << "\nBuilding Liouvillian..." << std::endl;

        try
        {
            build_liouvillian(hamiltonian, &Liouvillian);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }

        //---------------------------------------------------------------------------------

        std::cout << "\nCalculating the energy of the spin net at the initial Rho density matrix..." << std::endl;

        try
        {             
            QTensorNet::complexType energy = rho.ComputeOperatorTrace(&hamiltonian, 
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
        
        DoTask(thread_func_expectation_sigmaZ_rho, numThreadsMat, numSites);

        outFile << 0.0 << "\t";

        for (size_t j = 0UL; j < numSites; ++j) 
        {
            if(j < numSites-1) outFile << expectations[j][0] << "\t";
            else outFile << expectations[j][0] << std::endl;
        }

        //---------------------------------------------------------------------------------

        std::cout << "\nApplying unitary evolution operator and obtaining physical entities..." << std::endl; 

        QTensorNet::Integrators::TaylorSeriesIntegrator solver(1.0e-8);

        for(size_t iter = 0UL; iter < num_iter; ++iter)
        {
            std::cout << "Iteration: " << iter + 1UL << "/" << num_iter << "\r" << std::flush;

            try
            {
                rho.UpdateUsingTDVP(Liouvillian, solver, dt, 0UL, 3UL, true, false, 0UL, optimizer_attributes, 5);

                QTensorNet::complexType rho_tr = rho.ComputeOperatorTrace(nullptr, 
                                                                          0UL, 
                                                                          optimizer_attributes);

                rho *= QTensorNet::complexType(1.0, 0.0) / rho_tr;
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }

            DoTask(thread_func_expectation_sigmaZ_rho, numThreadsMat, numSites);

            outFile << static_cast<double>(iter + 1UL) * dt << "\t";

            for (size_t j = 0UL; j < numSites; ++j) 
            {
                if(j < numSites-1) outFile << expectations[j][iter + 1UL] << "\t";
                else outFile << expectations[j][iter + 1UL] << std::endl;
            }
        }

        outFile.close();

        std::cout << "\n\nCalculating the energy of the spin net at the final Rho density matrix..." << std::endl;

       try
        {             
            QTensorNet::complexType energy = rho.ComputeOperatorTrace(&hamiltonian, 
                                                                      0UL, 
                                                                      optimizer_attributes);

            std::cout << "Energy of the system: " << energy << std::endl;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }
        
        HANDLE_CUDA_ERROR(cudaFree(Sz_device));
        HANDLE_CUDA_ERROR(cudaFree(rhoC0_device));

        delete Liouvillian;

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