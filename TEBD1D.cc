#include <fstream>
#include <chrono>
#include <functional>

#include "TensorNetwork.hh"
#include "BasisGates.hh"
#include "CuArrayMethods.hh"
#include "PhysicalEntityCalculation.hh"

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

    size_t numSites = 10;
    int64_t physExtent = 2;
    int64_t maxVirtualExtent = 30;
    double absCutoff = 0.0;
    double relCutoff = 1.0e-5;
    size_t numThreads = 10;

    size_t num_iter = 700;
    double dt = 10.0 / static_cast<double>(num_iter);
    
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
        //mps.SetGlobalMode(true); // numThreads must be 1
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    } 

    //---Unit operator-------------------------------------------------------------------

    std::vector<std::vector<int64_t>> physExtentsOp(numSites, std::vector<int64_t>{physExtent, physExtent});

    QTensorNet::TensorNetwork unit_mpo(physExtentsOp, graph, root, maxVirtualExtent, numThreads);
    
    std::vector<std::vector<QTensorNet::complexType>> mpo_tensors_host;

    for (size_t i = 0; i < numSites; ++i)
    {
        std::vector<QTensorNet::complexType> data_host(unit_mpo.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

        data_host[0] = QTensorNet::complexType(1.0, 0.0);
        data_host[3] = QTensorNet::complexType(1.0, 0.0);

        mpo_tensors_host.push_back(data_host);
    }

    try
    {
        unit_mpo.SetState(mpo_tensors_host);
        unit_mpo.SetSVDConfig(absCutoff, relCutoff);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    }

    //---Hamiltonian-------------------------------------------------------------------

    std::cout << "\nBuilding Hamiltonian TN..." << std::endl;

    auto startH = std::chrono::steady_clock::now();

    QTensorNet::TensorNetwork hamiltonian_mpo(physExtentsOp, graph, root, maxVirtualExtent, numThreads);

    std::vector<std::vector<QTensorNet::complexType>> hamiltonian_mpo_tensors_host;

    for (size_t i = 0; i < numSites; ++i)
    {
        std::vector<QTensorNet::complexType> data_host(hamiltonian_mpo.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

        hamiltonian_mpo_tensors_host.push_back(data_host);
    }

    try
    {
        hamiltonian_mpo.SetState(hamiltonian_mpo_tensors_host);
        hamiltonian_mpo.SetSVDConfig(absCutoff, relCutoff);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    }
    
    auto sigmasum_host = QTensorNet::BasisGates::SigmaSum(hx, hy, hz);
    void* sigmasum_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmasum_host);

    auto sigmaIsigmaJsum_host = QTensorNet::BasisGates::SigmaISigmaJSum(Jx, Jy, Jz);
    void* sigmaIsigmaJsum_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaIsigmaJsum_host);

    for(size_t j = 0; j < numSites; ++j)
    {
        QTensorNet::TensorNetwork local_unit_mpo(unit_mpo);
        
        try
        {
            std::vector<int32_t> sigmasumModes = {local_unit_mpo.GetNode(j).physModes_[1], 
                                                  local_unit_mpo.GetNode(j).physModes_[1]};
            std::vector<int64_t> sigmasumExtents = {2, 2};

            local_unit_mpo.ApplySingleSiteGate(j, 
                                               sigmasum_device, 
                                               sigmasumModes,  
                                               sigmasumExtents);

            hamiltonian_mpo += local_unit_mpo;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }
    }

    for(size_t j = 0; j < numSites-1; ++j)
    {        
        QTensorNet::TensorNetwork local_unit_mpo(unit_mpo);
        
        try
        {
            std::vector<int32_t> sigmaIsigmaJsumModes = {local_unit_mpo.GetNode(j).physModes_[1], 
                                                         local_unit_mpo.GetNode(j + 1UL).physModes_[1], 
                                                         local_unit_mpo.GetNode(j).physModes_[1], 
                                                         local_unit_mpo.GetNode(j + 1UL).physModes_[1]};
            std::vector<int64_t> sigmaIsigmaJsumExtents = {2, 2, 2, 2};

            local_unit_mpo.ApplyTwoSiteGate(j, 
                                            j + 1UL, 
                                            sigmaIsigmaJsum_device, 
                                            sigmaIsigmaJsumModes, 
                                            sigmaIsigmaJsumExtents);
            
            hamiltonian_mpo += local_unit_mpo;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::exit(1);
        }
    }

    HANDLE_CUDA_ERROR(cudaFree(sigmasum_device));
    HANDLE_CUDA_ERROR(cudaFree(sigmaIsigmaJsum_device));

    auto finishH = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedH = finishH - startH;
    std::cout << "\nTotal spent time for Hamiltonian building: " << elapsedH.count() << " s." << std::endl;

    //---lambda functions--------------------------------------------------------------

    std::vector<std::vector<double>> expectations(numSites);
    std::vector<double> VonNeumannEntropy;
    std::vector<double> MutualInformation;

    auto exp_negIh1dtper2_host = QTensorNet::BasisGates::UnitarySigmaI(hx * dt / 2.0, hy * dt / 2.0, hz * dt / 2.0);
    void* exp_negIh1dtper2_device = QTensorNet::CuArrayMethods::VectorToGPUArray(exp_negIh1dtper2_host);

    auto exp_negIh2dt_host = QTensorNet::BasisGates::UnitarySigmaISigmaJ(Jx * dt, Jy * dt, Jz * dt);
    void* exp_negIh2dt_device = QTensorNet::CuArrayMethods::VectorToGPUArray(exp_negIh2dt_host);

    auto exp_negIh2dtper2_host = QTensorNet::BasisGates::UnitarySigmaISigmaJ(Jx * dt / 2.0, Jy * dt / 2.0, Jz * dt / 2.0);
    void* exp_negIh2dtper2_device = QTensorNet::CuArrayMethods::VectorToGPUArray(exp_negIh2dtper2_host);

    auto sigmaZ_host = QTensorNet::BasisGates::SigmaZ();
    void* sigmaZ_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaZ_host);

    auto thread_func_expectation_sigmaZ = [&mps, sigmaZ_device, &expectations](size_t thread_num, size_t start, size_t end) 
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
                                                            thread_num);
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

            double entropy = QTensorNet::PhysicalEntityCalculation::VonNeumannEntropy(rho_device, std::pow(2, keep_sites1.size()));

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

            double entropy1 = QTensorNet::PhysicalEntityCalculation::VonNeumannEntropy(rho_device1, 2);
            double entropy2 = QTensorNet::PhysicalEntityCalculation::VonNeumannEntropy(rho_device2, 2);
            double entropy12 = QTensorNet::PhysicalEntityCalculation::VonNeumannEntropy(rho_device12, 4);

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

    auto thread_func_exp_negIh1dtper2 = [&mps, exp_negIh1dtper2_device](size_t thread_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {            
            try
            {
                std::vector<int32_t> exp_negIh1dtper2Modes = {mps.GetNode(j).physModes_[0], mps.GetNode(j).physModes_[0]};
                std::vector<int64_t> exp_negIh1dtper2Extents = {2, 2};

                mps.ApplySingleSiteGate(j, 
                                        exp_negIh1dtper2_device, 
                                        exp_negIh1dtper2Modes,  
                                        exp_negIh1dtper2Extents,
                                        thread_num);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }
        }
    };

    auto thread_func_exp_negIh2dtper2 = [&mps, exp_negIh2dtper2_device](size_t thread_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {
            size_t node = 2UL * j;
            
            try
            {
                std::vector<int32_t> exp_negIh2dtper2Modes = {mps.GetNode(node).physModes_[0], 
                                                              mps.GetNode(node+1).physModes_[0], 
                                                              mps.GetNode(node).physModes_[0], 
                                                              mps.GetNode(node+1).physModes_[0]};
                std::vector<int64_t> exp_negIh2dtper2Extents = {2, 2, 2, 2};

                mps.ApplyTwoSiteGate(node, 
                                     node + 1, 
                                     exp_negIh2dtper2_device, 
                                     exp_negIh2dtper2Modes,
                                     exp_negIh2dtper2Extents, 
                                     thread_num);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }
        }
    };

    auto thread_func_exp_negIh2dt = [&mps, exp_negIh2dt_device](size_t thread_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {
            size_t node = 2UL * j + 1UL;
            
            try
            {
                std::vector<int32_t> exp_negIh2dtModes = {mps.GetNode(node).physModes_[0], 
                                                          mps.GetNode(node+1).physModes_[0], 
                                                          mps.GetNode(node).physModes_[0], 
                                                          mps.GetNode(node+1).physModes_[0]};
                std::vector<int64_t> exp_negIh2dtExtents = {2, 2, 2, 2};

                mps.ApplyTwoSiteGate(node, 
                                     node + 1, 
                                     exp_negIh2dt_device, 
                                     exp_negIh2dtModes, 
                                     exp_negIh2dtExtents,
                                     thread_num);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                std::exit(1);
            }
        }
    };

    //---------------------------------------------------------------------------------

    std::cout << "\nCalculating the energy of the spin net at the initial MPS state vector..." << std::endl;

    try
    {     
        QTensorNet::complexType energy = mps.ComputeMatrixElement(&hamiltonian_mpo);

        std::cout << "Energy of the system: " << energy << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    }

    DoTask(thread_func_expectation_sigmaZ, numThreads, numSites);

    calculate_entropy();

    calculate_mutual_information();

    std::cout << "\nApplying unitary evolution operator and obtaining physical entities..." << std::endl;

    for(size_t iter = 0; iter < num_iter; ++iter)
    {
        std::cout << "Iteration: " << iter + 1UL << "/" << num_iter << "\r" << std::flush;

        DoTask(thread_func_exp_negIh1dtper2, numThreads, numSites);

        DoTask(thread_func_exp_negIh2dtper2, numThreads, numSites / 2UL);

        DoTask(thread_func_exp_negIh2dt, numThreads, (numSites - 1UL) / 2UL);

        DoTask(thread_func_exp_negIh2dtper2, numThreads, numSites / 2UL);

        DoTask(thread_func_exp_negIh1dtper2, numThreads, numSites);

        auto [norm_device, descNorm] = mps.GetDensityMatrix();
        auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

        HANDLE_CUDA_ERROR(cudaFree(norm_device));

        mps *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);

        DoTask(thread_func_expectation_sigmaZ, numThreads, numSites);

        calculate_entropy();

        calculate_mutual_information();
    }
    
    std::ofstream outFile1("ExpectedSigmaZ_Local1DChain.txt");
    
    if (!outFile1.is_open()) 
    {
        std::cerr << "Error when creating expectations file." << std::endl;

        return 1;
    }

    for(size_t i = 0; i <= num_iter; ++i)
    {
        outFile1 << i * dt << "\t";
        
        for (size_t j = 0; j < numSites; ++j) 
        {
            if(j < numSites-1) outFile1 << expectations[j][i] << "\t";
            else outFile1 << expectations[j][i] << std::endl;
        }
    }

    outFile1.close();

    std::ofstream outFile2("VonNeumannEntropy_Local1DChain.txt");
    
    if (!outFile2.is_open()) 
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
    
    if (!outFile3.is_open()) 
    {
        std::cerr << "Error when creating MutualInformation file." << std::endl;

        return 1;
    }

    for(size_t i = 0; i <= num_iter; ++i)
    {
        outFile3 << i * dt << "\t" << MutualInformation[i] << std::endl;
    }

    outFile3.close();

    std::cout << "\n\nCalculating the energy of the spin net at the final MPS state vector..." << std::endl;

    try
    {            
        QTensorNet::complexType energy = mps.ComputeMatrixElement(&hamiltonian_mpo);

        std::cout << "Energy of the system: " << energy << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    }

    HANDLE_CUDA_ERROR(cudaFree(exp_negIh1dtper2_device));
    HANDLE_CUDA_ERROR(cudaFree(exp_negIh2dt_device));
    HANDLE_CUDA_ERROR(cudaFree(exp_negIh2dtper2_device));
    HANDLE_CUDA_ERROR(cudaFree(sigmaZ_device));
    
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "\nTotal spent time: " << elapsed.count() << " s." << std::endl;

    return 0;   
}