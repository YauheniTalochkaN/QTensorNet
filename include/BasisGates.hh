#pragma once

#include <vector>
#include <complex>

namespace QTensorNet
{
    namespace BasisGates 
    {
        using complexType = std::complex<double>;

        std::vector<complexType> SigmaX(double a = 1.0);
        std::vector<complexType> SigmaY(double a = 1.0);
        std::vector<complexType> SigmaZ(double a = 1.0);
        std::vector<complexType> SigmaMinus(double a = 1.0);
        std::vector<complexType> SigmaPlus(double a = 1.0);
        std::vector<complexType> SigmaSum(double hx, double hy, double hz);

        std::vector<complexType> CNOT();
        std::vector<complexType> Hadamard();

        std::vector<complexType> UnitarySigmaI(double hx, double hy, double hz);
        std::vector<complexType> UnitarySigmaISigmaJ(double Jx, double Jy, double Jz);
        std::vector<complexType> SigmaISigmaJSum(double Jx, double Jy, double Jz);
    }
}