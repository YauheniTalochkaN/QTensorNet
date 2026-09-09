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
    int64_t maxVirtualExtent = 40;
    double absCutoff = 0.0;
    double relCutoff = 1.0e-5;
    size_t numThreads = 1;
    
    size_t num_iter = 700;
    double dt = 10.0 / static_cast<double>(num_iter);

    double hx = -0.5, hy = 0.3, hz = 1.6;
    double Jx = 0.7, Jy = -0.5, Jz = -1.2;

    size_t length = 10;
    double alpha = 3.0;

    std::vector<size_t> keep_sites1 = {0, 1, 2, 3, 4};
    std::vector<size_t> keep_sites2 = {1, 5};

    std::vector<std::vector<int64_t>> physExtentsVec(numSites, std::vector<int64_t>{physExtent});

    QTensorNet::virtualModesGraphType graph;

    size_t root = numSites / 2UL;

    for(size_t i = 0; i < numSites-1; ++i)
    {
        graph.insert(std::make_tuple(i, i+1UL, 1L));
    }

    QTensorNet::TensorNetwork psi(physExtentsVec, graph, root, maxVirtualExtent, numThreads);

    std::vector<std::vector<QTensorNet::complexType>> psi_tensors_host;

    std::cout << "\nInitializing Psi tensors..." << std::endl;

    for(size_t i = 0; i < numSites; ++i)
    {        
        std::vector<QTensorNet::complexType> data_host(psi.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

        data_host[1] = QTensorNet::complexType(1.0, 0.0);

        /* if(i % 2 == 0) 
        {
            data_host[1] = QTensorNet::complexType(1.0, 0.0);
        }
        else
        {
            data_host[0] = QTensorNet::complexType(1.0, 0.0);
        } */

        psi_tensors_host.push_back(data_host);

        std::cout << "Site " << i << ": tensor[0] = (" << psi_tensors_host[i][0].real() 
                  << ", " << psi_tensors_host[i][0].imag() << "), tensor[1] = (" 
                  << psi_tensors_host[i][1].real() << ", " << psi_tensors_host[i][1].imag() << ")" << std::endl;
    }

    try
    {
        psi.SetState(psi_tensors_host);
        psi.SetSVDConfig(absCutoff, relCutoff);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    } 

    //---Unit operator-------------------------------------------------------------------

    std::vector<std::vector<int64_t>> physExtentsOp(numSites, std::vector<int64_t>{physExtent, physExtent});

    QTensorNet::TensorNetwork unit_mpo(physExtentsOp, graph, root, maxVirtualExtent, numThreads);
    
    std::vector<std::vector<QTensorNet::complexType>> unit_mpo_tensors_host;

    for (size_t i = 0; i < numSites; ++i)
    {
        std::vector<QTensorNet::complexType> data_host(unit_mpo.GetTensorSize(i), QTensorNet::complexType(0.0, 0.0));

        data_host[0] = QTensorNet::complexType(1.0, 0.0);
        data_host[3] = QTensorNet::complexType(1.0, 0.0);

        unit_mpo_tensors_host.push_back(data_host);
    }

    try
    {
        unit_mpo.SetState(unit_mpo_tensors_host);
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

    auto sigmaX_host = QTensorNet::BasisGates::SigmaX();
    void* sigmaX_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaX_host);

    auto sigmaY_host = QTensorNet::BasisGates::SigmaY();
    void* sigmaY_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaY_host);

    auto sigmaZ_host = QTensorNet::BasisGates::SigmaZ();
    void* sigmaZ_device = QTensorNet::CuArrayMethods::VectorToGPUArray(sigmaZ_host);

    for(size_t j = 0; j < numSites; ++j)
    {               
        try
        {
            QTensorNet::TensorNetwork local_unit_mpo(unit_mpo);

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

    for(size_t i = 0; i < numSites; ++i)
    {
        for(size_t j = i + 1UL; j < numSites; ++j)
        {           
            double l = std::fabs(static_cast<double>(i) - static_cast<double>(j));
            
            if(static_cast<size_t>(l) <= length)
            {
                try
                {
                    QTensorNet::TensorNetwork local_unit_mpo(unit_mpo);

                    std::vector<int32_t> sigmaIModes = {local_unit_mpo.GetNode(i).physModes_[1],          
                                                        local_unit_mpo.GetNode(i).physModes_[1]};

                    std::vector<int32_t> sigmaJModes = {local_unit_mpo.GetNode(j).physModes_[1],          
                                                        local_unit_mpo.GetNode(j).physModes_[1]};

                    std::vector<int64_t> sigmaExtents = {2, 2};

                    local_unit_mpo.ApplySingleSiteGate(i, sigmaX_device, sigmaIModes, sigmaExtents);
                    local_unit_mpo.ApplySingleSiteGate(j, sigmaX_device, sigmaJModes, sigmaExtents);

                    local_unit_mpo *= QTensorNet::complexType(Jx / std::pow(l, alpha), 0.0);
                    
                    hamiltonian_mpo += local_unit_mpo;
                    
                    local_unit_mpo = unit_mpo;

                    local_unit_mpo.ApplySingleSiteGate(i, sigmaY_device, sigmaIModes, sigmaExtents);
                    local_unit_mpo.ApplySingleSiteGate(j, sigmaY_device, sigmaJModes, sigmaExtents);

                    local_unit_mpo *= QTensorNet::complexType(Jy / std::pow(l, alpha), 0.0);
                    
                    hamiltonian_mpo += local_unit_mpo;

                    local_unit_mpo = unit_mpo;

                    local_unit_mpo.ApplySingleSiteGate(i, sigmaZ_device, sigmaIModes, sigmaExtents);
                    local_unit_mpo.ApplySingleSiteGate(j, sigmaZ_device, sigmaJModes, sigmaExtents);

                    local_unit_mpo *= QTensorNet::complexType(Jz / std::pow(l, alpha), 0.0);
                    
                    hamiltonian_mpo += local_unit_mpo;
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    std::exit(1);
                }
            }
        }
    }

    HANDLE_CUDA_ERROR(cudaFree(sigmasum_device));
    HANDLE_CUDA_ERROR(cudaFree(sigmaX_device));
    HANDLE_CUDA_ERROR(cudaFree(sigmaY_device));

    auto finishH = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedH = finishH - startH;
    std::cout << "\nTotal spent time for Hamiltonian building: " << elapsedH.count() << " s." << std::endl;

    //---lambda functions for observables----------------------------------------------

    std::vector<std::vector<double>> expectations(numSites);
    std::vector<double> VonNeumannEntropy;
    std::vector<double> MutualInformation;

    auto thread_func_expectation_sigmaZ = [&psi, sigmaZ_device, &expectations](size_t thread_num, size_t start, size_t end) 
    {
        for(size_t j = start; j < end; ++j)
        {           
            QTensorNet::complexType expectationValue(0.0, 0.0);
            
            try
            {
                std::vector<int32_t> sigmaZModes = {psi.GetNode(j).physModes_[0], 
                                                    psi.GetNode(j).physModes_[0]};
                std::vector<int64_t> sigmaZExtents = {2, 2};
            
                expectationValue = psi.ComputeMatrixElement(sigmaZ_device, 
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

    auto calculate_entropy = [&psi, &keep_sites1, &VonNeumannEntropy]() 
    {
        try
        {      
            auto [rho_device, descRho] = psi.GetDensityMatrix(keep_sites1);

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

    auto calculate_mutual_information = [&psi, &keep_sites2, &MutualInformation]() 
    {
        try
        {      
            auto [rho_device1, descRho1] = psi.GetDensityMatrix({keep_sites2[0]});
            auto [rho_device2, descRho2] = psi.GetDensityMatrix({keep_sites2[1]});
            auto [rho_device12, descRho12] = psi.GetDensityMatrix(keep_sites2);

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

    //---------------------------------------------------------------------------------

    std::cout << "\nCalculating the energy of the spin net at the initial Psi state vector..." << std::endl;

    try
    {     
        QTensorNet::complexType energy = psi.ComputeMatrixElement(&hamiltonian_mpo);

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
            
        QTensorNet::TensorNetwork k1(psi);
        k1 *= hamiltonian_mpo;

        QTensorNet::TensorNetwork k2(k1);
        k2 *= QTensorNet::complexType(0.0, -dt / 2.0);
        k2 += psi;
        k2 *= hamiltonian_mpo;

        QTensorNet::TensorNetwork k3(k2);
        k3 *= QTensorNet::complexType(0.0, -dt / 2.0);
        k3 += psi;
        k3 *= hamiltonian_mpo;

        QTensorNet::TensorNetwork k4(k3);
        k4 *= QTensorNet::complexType(0.0, -dt);
        k4 += psi;
        k4 *= hamiltonian_mpo;

        k2 *= QTensorNet::complexType(2.0, 0.0);
        k3 *= QTensorNet::complexType(2.0, 0.0);

        k1 += k2;
        k1 += k3;
        k1 += k4;

        k2.ClearNet();
        k3.ClearNet();
        k4.ClearNet();

        k1 *= QTensorNet::complexType(0.0, -dt / 6.0);

        psi += k1;

        k1.ClearNet();

        auto [norm_device, descNorm] = psi.GetDensityMatrix();
        auto norm_host = QTensorNet::CuArrayMethods::GPUArrayToVector(norm_device, 1).at(0);

        HANDLE_CUDA_ERROR(cudaFree(norm_device));

        psi *= QTensorNet::complexType(1.0, 0.0) / std::sqrt(norm_host);

        DoTask(thread_func_expectation_sigmaZ, numThreads, numSites);

        calculate_entropy();

        calculate_mutual_information();
    }
    
    std::ofstream outFile1("ExpectedSigmaZ_1DChain.txt");
    
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

    std::ofstream outFile2("VonNeumannEntropy_1DChain.txt");
    
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

    std::ofstream outFile3("MutualInformation_1DChain.txt");
    
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

    std::cout << "\n\nCalculating the energy of the spin net at the final Psi state vector..." << std::endl;

    try
    {            
        QTensorNet::complexType energy = psi.ComputeMatrixElement(&hamiltonian_mpo);

        std::cout << "Energy of the system: " << energy << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::exit(1);
    }

    HANDLE_CUDA_ERROR(cudaFree(sigmaZ_device));
    
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "\nTotal spent time: " << elapsed.count() << " s." << std::endl;

    return 0;   
}