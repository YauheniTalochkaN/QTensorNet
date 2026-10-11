#include <fstream>
#include <chrono>
#include <functional>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"
#include "CuOperatorMethods.hh"
#include "ThreadPool.hh"

void DoTask(QTensorNet::ThreadPool& pool, 
            const std::function<void(size_t, size_t, size_t)>& task, 
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
            futures.emplace_back(pool.AddTask(task, st, start, end));
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
    auto start = std::chrono::steady_clock::now();
    
    const size_t cuTensornetVersion = cutensornetGetVersion();
    std::cout << "cuTensorNet-vers: " << cuTensornetVersion << std::endl;

    cudaDeviceProp prop;
    int deviceId{-1};
    HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    std::cout << "GPU-local-id:" << deviceId << "\n"
              << "GPU-name:" << prop.name << std::endl;

    //QTensorNet::TensorNetwork::check_ = false;

    size_t numSites = 10UL;
    int64_t physExtent = 2L;
    int64_t maxVirtualExtent = 30L;
    double absCutoff = 0.0;
    double relCutoff = 1.0e-8;
    size_t numThreads = 10UL;
    size_t STorder = 3UL;

    size_t num_iter = 500UL;
    double dt = 10.0 / static_cast<double>(num_iter);

    QTensorNet::ThreadPool pool(numThreads);
    
    std::vector<size_t> keep_sites1 = {0, 1, 2, 3, 4};
    std::vector<size_t> keep_sites2 = {1, 5};

    double hx = -0.5, hy = 0.3, hz = 1.6;
    double Jx = 0.7, Jy = -0.5, Jz = -1.2;

    std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

    QTensorNet::virtualModesGraphType graph;

    size_t root = numSites / 2UL;
    
    for(size_t i = 0; i < numSites-1; ++i)
    {
        graph.insert(std::make_tuple(i, i+1UL, 1L));
    }

    QTensorNet::TensorNetwork mps(physExtentsVec, graph, root, maxVirtualExtent, numThreads);

    std::vector<std::vector<QTensorNet::complexType>> mps_tensors_host;

    std::cout << "\nInitializing MPS tensors for initial state..." << std::endl;

    for(size_t i = 0; i < numSites; ++i)
    {
        std::vector<QTensorNet::complexType> data_host(mps.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

        data_host[1] = QTensorNet::complexType(1.0, 0.0);

        /* if(i % 2 == 0) 
        {
            data_host[1] = QTensorNet::complexType(1.0, 0.0);
        }
        else
        {
            data_host[0] = QTensorNet::complexType(1.0, 0.0);
        } */

        mps_tensors_host.push_back(data_host);

        std::cout << "Site " << i << ": tensor[0] = (" << mps_tensors_host[i][0].real() 
                  << ", " << mps_tensors_host[i][0].imag() << "), tensor[1] = (" 
                  << mps_tensors_host[i][1].real() << ", " << mps_tensors_host[i][1].imag() << ")" << std::endl;
    }

    try
    {
        mps.SetState(mps_tensors_host);
        mps.SetSVDConfig(absCutoff, relCutoff);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    } 

    //---lambda functions--------------------------------------------------------------

    std::vector<std::vector<double>> expectations(numSites);
    std::vector<double> VonNeumannEntropy;
    std::vector<double> MutualInformation;

    auto pk_vec = QTensorNet::GetSuzukiCoeffs(STorder);

    std::vector<void*> exp_negIh1pkdt_device(pk_vec.size(), nullptr);
    std::vector<void*> exp_negIh2pkdtper2_device(pk_vec.size(), nullptr);

    for(size_t s = 0UL; s < pk_vec.size(); ++s)
    {
        exp_negIh1pkdt_device[s] = QTensorNet::CuArrayMethods::VectorToGPUArray(
            QTensorNet::BasisGates::UnitarySigmaI(pk_vec[s] * hx * dt, pk_vec[s] * hy * dt, pk_vec[s] * hz * dt));

        exp_negIh2pkdtper2_device[s] = QTensorNet::CuArrayMethods::VectorToGPUArray(
            QTensorNet::BasisGates::UnitarySigmaISigmaJ(pk_vec[s] * Jx * dt / 2.0, pk_vec[s] * Jy * dt / 2.0, pk_vec[s] * Jz * dt / 2.0));
    }

    auto sigmaZ_host = QTensorNet::BasisGates::SigmaZ();
    void* sigmaZ_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaZ_host);

    auto thread_func_expectation_sigmaZ = [&mps, sigmaZ_device, &expectations](size_t stream_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {           
            QTensorNet::complexType expectationValue(0.0, 0.0);
            
            try
            {
                std::vector<int32_t> sigmaZModes = {mps.GetNode(j).physModes_[0], 
                                                    mps.GetNode(j).physModes_[0]};
                std::vector<int64_t> sigmaZExtents = {2, 2};
            
                expectationValue = mps.ComputeMatrixElement(sigmaZ_device, 
                                                            sigmaZModes,  
                                                            sigmaZExtents,
                                                            nullptr,
                                                            stream_num);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }

            expectations[j].push_back(expectationValue.real());
        }
    };

    auto calculate_entropy = [&mps, &keep_sites1, &VonNeumannEntropy]() 
    {
        try
        {      
            auto [rho_device, descRho] = mps.GetDensityMatrix(keep_sites1);

            double entropy = QTensorNet::CuOperatorMethods::VonNeumannEntropy(rho_device, std::pow(2, keep_sites1.size()));

            VonNeumannEntropy.emplace_back(entropy);

            HANDLE_CUDA_ERROR(cudaFree(rho_device));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }
    };

    auto calculate_mutual_information = [&mps, &keep_sites2, &MutualInformation]() 
    {
        try
        {      
            auto [rho_device1, descRho1] = mps.GetDensityMatrix({keep_sites2[0]});
            auto [rho_device2, descRho2] = mps.GetDensityMatrix({keep_sites2[1]});
            auto [rho_device12, descRho12] = mps.GetDensityMatrix(keep_sites2);

            double entropy1 = QTensorNet::CuOperatorMethods::VonNeumannEntropy(rho_device1, 2);
            double entropy2 = QTensorNet::CuOperatorMethods::VonNeumannEntropy(rho_device2, 2);
            double entropy12 = QTensorNet::CuOperatorMethods::VonNeumannEntropy(rho_device12, 4);

            MutualInformation.emplace_back(entropy1 + entropy2 - entropy12);

            HANDLE_CUDA_ERROR(cudaFree(rho_device1));
            HANDLE_CUDA_ERROR(cudaFree(rho_device2));
            HANDLE_CUDA_ERROR(cudaFree(rho_device12));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }
    };

    auto thread_func_exp_negIh1dt = [&mps, &exp_negIh1pkdt_device](size_t STstep, size_t stream_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {            
            try
            {
                std::vector<int32_t> Modes = {mps.GetNode(j).physModes_[0], 
                                              mps.GetNode(j).physModes_[0]};
                std::vector<int64_t> Extents = {2, 2};

                mps.ApplySingleSiteGate(j, 
                                        exp_negIh1pkdt_device[STstep], 
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

    auto thread_func_exp_negIh2dtper2_even = [&mps, &exp_negIh2pkdtper2_device](size_t STstep, size_t stream_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {
            size_t node = 2UL * j;
            
            try
            {
                std::vector<int32_t> Modes = {mps.GetNode(node).physModes_[0], 
                                              mps.GetNode(node+1).physModes_[0], 
                                              mps.GetNode(node).physModes_[0], 
                                              mps.GetNode(node+1).physModes_[0]};
                std::vector<int64_t> Extents = {2, 2, 2, 2};

                mps.ApplyTwoSiteGate(node, 
                                     node + 1, 
                                     exp_negIh2pkdtper2_device[STstep], 
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

    auto thread_func_exp_negIh2dtper2_odd = [&mps, &exp_negIh2pkdtper2_device](size_t STstep, size_t stream_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {
            size_t node = 2UL * j + 1UL;
            
            try
            {
                std::vector<int32_t> Modes = {mps.GetNode(node).physModes_[0], 
                                              mps.GetNode(node+1).physModes_[0], 
                                              mps.GetNode(node).physModes_[0], 
                                              mps.GetNode(node+1).physModes_[0]};
                std::vector<int64_t> Extents = {2, 2, 2, 2};

                mps.ApplyTwoSiteGate(node, 
                                     node + 1, 
                                     exp_negIh2pkdtper2_device[STstep], 
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

    DoTask(pool, thread_func_expectation_sigmaZ, numThreads, numSites);

    calculate_entropy();

    calculate_mutual_information();

    for(size_t iter = 0; iter < num_iter; ++iter)
    {
        std::cout << "Iteration: " << iter + 1UL << "/" << num_iter << "\r" << std::flush;

        for(size_t s = 0UL; s < pk_vec.size(); ++s)
        {
            DoTask(pool, thread_func_exp_negIh2dtper2_even, s, numThreads, numSites / 2UL);
            DoTask(pool, thread_func_exp_negIh2dtper2_odd, s, numThreads, (numSites - 1UL) / 2UL);

            DoTask(pool, thread_func_exp_negIh1dt, s, numThreads, numSites);

            DoTask(pool, thread_func_exp_negIh2dtper2_odd, s, numThreads, (numSites - 1UL) / 2UL);
            DoTask(pool, thread_func_exp_negIh2dtper2_even, s, numThreads, numSites / 2UL);
        }

        auto [norm_device, descNorm] = mps.GetDensityMatrix();
        auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

        HANDLE_CUDA_ERROR(cudaFree(norm_device));

        mps *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);

        DoTask(pool, thread_func_expectation_sigmaZ, numThreads, numSites);

        calculate_entropy();

        calculate_mutual_information();
    }
    
    std::ofstream outFile1("ExpectedSigmaZ_Local1DChain.txt");
    
    if(!outFile1.is_open()) 
    {
        std::cerr << "Error when creating expectations file." << std::endl;

        return 1;
    }

    for(size_t i = 0; i <= num_iter; ++i)
    {
        outFile1 << i * dt << "\t";
        
        for(size_t j = 0; j < numSites; ++j) 
        {
            if(j < numSites-1) outFile1 << expectations[j][i] << "\t";
            else outFile1 << expectations[j][i] << std::endl;
        }
    }

    outFile1.close();

    std::ofstream outFile2("VonNeumannEntropy_Local1DChain.txt");
    
    if(!outFile2.is_open()) 
    {
        std::cerr << "Error when creating VonNeumannEntropy file." << std::endl;

        return 1;
    }

    for(size_t i = 0; i <= num_iter; ++i)
    {
        outFile2 << i * dt << "\t" << VonNeumannEntropy[i] << std::endl;
    }

    outFile2.close();

    std::ofstream outFile3("MutualInformation_Local1DChain.txt");
    
    if(!outFile3.is_open()) 
    {
        std::cerr << "Error when creating MutualInformation file." << std::endl;

        return 1;
    }

    for(size_t i = 0; i <= num_iter; ++i)
    {
        outFile3 << i * dt << "\t" << MutualInformation[i] << std::endl;
    }

    outFile3.close();

    for(auto it : exp_negIh1pkdt_device)
    {
        HANDLE_CUDA_ERROR(cudaFree(it));
    }
    
    for(auto it : exp_negIh2pkdtper2_device)
    {
        HANDLE_CUDA_ERROR(cudaFree(it));
    }

    HANDLE_CUDA_ERROR(cudaFree(sigmaZ_device));
    
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "\nTotal spent time: " << elapsed.count() << " s." << std::endl;

    return 0;   
}