#include <fstream>
#include <chrono>
#include <functional>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"
#include "CuOperatorMethods.hh"
#include "ThreadPool.hh"

std::vector<std::tuple<size_t, size_t, size_t>> SquareLattice(size_t Nx, size_t Ny)
{
    size_t Nbond = 0;

    if(Nx > 0) Nbond += (Nx - 1) * Ny;
    if(Ny > 0) Nbond += Nx * (Ny - 1);

    std::vector<std::tuple<size_t, size_t, size_t>> latt;
    latt.reserve(Nbond);

    for (size_t i = 0; i < Nx; ++i)
    {
        for (size_t j = 0; j < Ny; ++j)
        {
            size_t site = i + j * Nx;

            if (i + 1 < Nx)
            {
                latt.emplace_back(site, (i + 1) + j * Nx, i % 2);
            }

            if (j + 1 < Ny)
            {
                latt.emplace_back(site, i + (j + 1) * Nx, 2 + (j % 2));
            }
        }
    }

    if (latt.size() != Nbond)
    {
        std::cerr << "SquareLattice: Wrong number of bonds." << std::endl;
    }

    return latt;
}

void DoTask(QTensorNet::ThreadPool& pool, 
            const std::function<void(const std::vector<std::pair<size_t, size_t>>&, size_t, size_t, size_t, size_t)>& task,
            const std::vector<std::pair<size_t, size_t>>& pairs, 
            size_t STstep, size_t stream_num, size_t num_sites)
{
    size_t chunk_size = (num_sites + stream_num - 1) / stream_num;

    std::vector<std::future<void>> futures;

    for(size_t st = 0; st < stream_num; ++st)
    {
        size_t start = st * chunk_size;
        size_t end = std::min(start + chunk_size, num_sites);

        if(start < num_sites)
        {
            futures.emplace_back(pool.AddTask(task, pairs, STstep, st, start, end));
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

void DoTask(QTensorNet::ThreadPool& pool, 
            const std::function<void(size_t, size_t, size_t, size_t)>& task, 
            size_t STstep, size_t stream_num, size_t num_sites)
{
    size_t chunk_size = (num_sites + stream_num - 1) / stream_num;

    std::vector<std::future<void>> futures;

    for(size_t st = 0; st < stream_num; ++st)
    {
        size_t start = st * chunk_size;
        size_t end = std::min(start + chunk_size, num_sites);

        if(start < num_sites)
        {
            futures.emplace_back(pool.AddTask(task, STstep, st, start, end));
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

        size_t Nx = 8UL;
        size_t Ny = 8UL;
        size_t numSites = Nx * Ny;
        int64_t physExtent = 2L;
        int64_t maxVirtualExtent = 8L;
        double absCutoff = 0.0;
        double relCutoff = 1.0e-8;
        size_t workSpaceLimit = 60UL * 1024UL;
        size_t STorder = 3UL;

        QTensorNet::CuTensorNetMethods::ContractionOptimizerAttributes optimizer_attributes = 
        {{CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES, 100},
         {CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS, 10000}};
        
        size_t num_iter = 100UL;
        double tmax = 1.0;
        double hz = 3.04438;

        double dt = tmax / static_cast<double>(num_iter);

        auto latt = SquareLattice(Nx, Ny);

        size_t numThreads = 20UL;
        QTensorNet::ThreadPool pool(numThreads);

        std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

        QTensorNet::virtualModesGraphType graph;

        size_t root = 0UL;

        std::array<std::vector<std::pair<size_t, size_t>>, 2UL> hpairs, vpairs;
        
        for(const auto& [i, j, b] : latt)
        {        
            graph.insert(std::make_tuple(i, j, 1L));

            switch(b)
            {
                case 0UL:
                    hpairs[0].emplace_back(i, j);
                    break;
                case 1UL:
                    hpairs[1].emplace_back(i, j);
                    break;
                case 2UL:
                    vpairs[0].emplace_back(i, j);
                    break;
                case 3UL:
                    vpairs[1].emplace_back(i, j);
                    break;
                default:
                    break;
            }
        }

        QTensorNet::TensorNetwork peps(physExtentsVec, graph, root, maxVirtualExtent, 1UL, workSpaceLimit);

        std::vector<std::vector<QTensorNet::complexType>> peps_tensors_host;

        std::cout << "\nInitializing PEPS tensors for initial state..." << std::endl;

        for(size_t i = 0; i < numSites; ++i)
        {
            std::vector<QTensorNet::complexType> data_host(peps.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

            data_host[0] = QTensorNet::complexType(1.0, 0.0);

            peps_tensors_host.push_back(data_host);

            std::cout << "Site " << i << ": tensor[0] = (" << peps_tensors_host[i][0].real() 
                      << ", " << peps_tensors_host[i][0].imag() << "), tensor[1] = (" 
                      << peps_tensors_host[i][1].real() << ", " << peps_tensors_host[i][1].imag() << ")" << std::endl;
        }

        try
        {
            peps.SetState(peps_tensors_host);
            peps.SetSVDConfig(absCutoff, relCutoff);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        } 

        //---lambda functions--------------------------------------------------------------

        std::vector<std::vector<double>> expectations(numSites);

        auto pk_vec = QTensorNet::GetSuzukiCoeffs(STorder);

        std::vector<void*> exp_negIzpkdt_device(pk_vec.size(), nullptr);
        std::vector<void*> exp_negIxxpkdtper2_device(pk_vec.size(), nullptr);

        for(size_t s = 0UL; s < pk_vec.size(); ++s)
        {
            exp_negIzpkdt_device[s] = QTensorNet::CuArrayMethods::VectorToGPUArray(
                QTensorNet::BasisGates::UnitarySigmaI(0.0, 0.0, pk_vec[s] * hz * dt));

            exp_negIxxpkdtper2_device[s] = QTensorNet::CuArrayMethods::VectorToGPUArray(
                QTensorNet::BasisGates::UnitarySigmaISigmaJ(pk_vec[s] * dt / 2.0, 0.0, 0.0));
        }

        auto sigmaZ_host = QTensorNet::BasisGates::SigmaZ();
        void* sigmaZ_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaZ_host);

        auto eval_expectation_sigmaZ_psi = [&peps, sigmaZ_device, &expectations, &optimizer_attributes, &numSites]() 
        {
            for(size_t j = 0; j < numSites; ++j)
            {           
                QTensorNet::complexType expectationValue(0.0, 0.0);

                try
                {
                    std::vector<int32_t> sigmaZModes = {peps.GetNode(j).physModes_[0], 
                                                        peps.GetNode(j).physModes_[0]};
                    std::vector<int64_t> sigmaZExtents = {2, 2};
                    
                    expectationValue = peps.ComputeMatrixElement(sigmaZ_device, 
                                                                 sigmaZModes,  
                                                                 sigmaZExtents,
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

        auto thread_func_exp_negIzdt = [&peps, &exp_negIzpkdt_device](size_t STstep, size_t stream_num, size_t start, size_t end) 
        {
            for(size_t j = start; j < end; ++j)
            {            
                try
                {
                    std::vector<int32_t> Modes = {peps.GetNode(j).physModes_[0], 
                                                  peps.GetNode(j).physModes_[0]};
                    std::vector<int64_t> Extents = {2, 2};

                    peps.ApplySingleSiteGate(j, 
                                             exp_negIzpkdt_device[STstep], 
                                             Modes,  
                                             Extents,
                                             stream_num);
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }
            }
        };

        auto thread_func_exp_negIhxxdtper2 = [&peps, &exp_negIxxpkdtper2_device]
        (const std::vector<std::pair<size_t, size_t>>& pairs, size_t STstep, size_t stream_num, size_t start, size_t end) 
        {
            for(size_t k = start; k < end; ++k)
            {
                auto [i, j] = pairs[k];

                try
                {
                    std::vector<int32_t> Modes = {peps.GetNode(i).physModes_[0], 
                                                  peps.GetNode(j).physModes_[0], 
                                                  peps.GetNode(i).physModes_[0], 
                                                  peps.GetNode(j).physModes_[0]};
                    std::vector<int64_t> Extents = {2, 2, 2, 2};

                    peps.ApplyTwoSiteGate(i, 
                                          j, 
                                          exp_negIxxpkdtper2_device[STstep], 
                                          Modes,
                                          Extents, 
                                          stream_num);
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }
            }
        };

        //---------------------------------------------------------------------------------

        std::cout << "\nApplying unitary evolution operator and obtaining physical entities..." << std::endl;

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

        for(size_t iter = 0; iter < num_iter; ++iter)
        {
            std::cout << "Iteration: " << iter + 1UL << "/" << num_iter << "\r" << std::flush;

            QTensorNet::CuTensorNetMethods::MPI_ = false;

            peps.SetNumStreams(numThreads);

            for(size_t s = 0UL; s < pk_vec.size(); ++s)
            {
                DoTask(pool, thread_func_exp_negIhxxdtper2, hpairs[0], s, numThreads, hpairs[0].size());
                DoTask(pool, thread_func_exp_negIhxxdtper2, hpairs[1], s, numThreads, hpairs[1].size());

                DoTask(pool, thread_func_exp_negIhxxdtper2, vpairs[0], s, numThreads, vpairs[0].size());
                DoTask(pool, thread_func_exp_negIhxxdtper2, vpairs[1], s, numThreads, vpairs[1].size());

                DoTask(pool, thread_func_exp_negIzdt, s, numThreads, numSites);

                DoTask(pool, thread_func_exp_negIhxxdtper2, vpairs[0], s, numThreads, vpairs[0].size());
                DoTask(pool, thread_func_exp_negIhxxdtper2, vpairs[1], s, numThreads, vpairs[1].size());

                DoTask(pool, thread_func_exp_negIhxxdtper2, hpairs[0], s, numThreads, hpairs[0].size());
                DoTask(pool, thread_func_exp_negIhxxdtper2, hpairs[1], s, numThreads, hpairs[1].size());
            }

            peps.SetNumStreams(1UL);

            if(numProcs > 1)
            {
                QTensorNet::CuTensorNetMethods::MPI_ = true;
            }

            auto [norm_device, descNorm] = peps.GetDensityMatrix({}, true, 0UL, optimizer_attributes);
            auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

            HANDLE_CUDA_ERROR(cudaFree(norm_device));

            peps *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);

            eval_expectation_sigmaZ_psi();

            outFile << static_cast<double>(iter + 1UL) * dt << "\t";

            for(size_t j = 0UL; j < numSites; ++j) 
            {
                if(j < numSites-1) outFile << expectations[j][iter + 1UL] << "\t";
                else outFile << expectations[j][iter + 1UL] << std::endl;
            }
        }

        outFile.close();

        for(auto it : exp_negIzpkdt_device)
        {
            HANDLE_CUDA_ERROR(cudaFree(it));
        }

        for(auto it : exp_negIxxpkdtper2_device)
        {
            HANDLE_CUDA_ERROR(cudaFree(it));
        }

        HANDLE_CUDA_ERROR(cudaFree(sigmaZ_device));

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