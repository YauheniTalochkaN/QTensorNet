#include "BasisGates.hh"

namespace QTensorNet
{
    namespace BasisGates 
    {
        std::vector<complexType> SigmaX(double a) 
        {
            std::vector<complexType> result(4);
        
            result[0] = complexType(0.0, 0.0);
            result[1] = complexType(a, 0.0);
            result[2] = complexType(a, 0.0);
            result[3] = complexType(0.0, 0.0);
        
            return result;
        };

        std::vector<complexType> SigmaY(double a) 
        {
            std::vector<complexType> result(4);
        
            result[0] = complexType(0.0, 0.0);
            result[1] = complexType(0.0, -a);
            result[2] = complexType(0.0, a);
            result[3] = complexType(0.0, 0.0);
        
            return result;
        }

        std::vector<complexType> SigmaZ(double a) 
        {
            std::vector<complexType> result(4);
        
            result[0] = complexType(a, 0.0);
            result[1] = complexType(0.0, 0.0);
            result[2] = complexType(0.0, 0.0);
            result[3] = complexType(-a, 0.0);
        
            return result;
        }

        std::vector<complexType> SigmaMinus(double a)
        {
            std::vector<complexType> result(4);
        
            result[0] = complexType(0.0, 0.0);
            result[1] = complexType(0.0, 0.0);
            result[2] = complexType(2.0 * a, 0.0);
            result[3] = complexType(0.0, 0.0);
        
            return result;
        }

        std::vector<complexType> SigmaPlus(double a)
        {
            std::vector<complexType> result(4);
        
            result[0] = complexType(0.0, 0.0);
            result[1] = complexType(2.0 * a, 0.0);
            result[2] = complexType(0.0, 0.0);
            result[3] = complexType(0.0, 0.0);
        
            return result;
        }

        std::vector<complexType> SigmaSum(double hx, double hy, double hz)
        {
            std::vector<complexType> result(4);

            result[0] = complexType(hz, 0.0);
            result[1] = complexType(hx, -hy);
            result[2] = complexType(hx, hy);
            result[3] = complexType(-hz, 0.0);
        
            return result;
        }

        std::vector<complexType> CNOT() 
        {
            std::vector<complexType> result(16, complexType(0.0, 0.0));
        
            result[0]  = complexType(1.0, 0.0);
            result[7]  = complexType(1.0, 0.0);
            result[10] = complexType(1.0, 0.0);
            result[13] = complexType(1.0, 0.0);
        
            return result;
        }

        std::vector<complexType> Hadamard() 
        {
            std::vector<complexType> result(4);
        
            const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        
            result[0] = complexType(inv_sqrt2, 0.0);
            result[1] = complexType(inv_sqrt2, 0.0);
            result[2] = complexType(inv_sqrt2, 0.0);
            result[3] = complexType(-inv_sqrt2, 0.0);
        
            return result;
        }

        std::vector<complexType> UnitarySigmaI(double hx, double hy, double hz)
        {
            std::vector<complexType> result(4);
        
            double h = std::sqrt(hx*hx + hy*hy + hz*hz);
        
            result[0] = complexType(std::cos(h), -hz * std::sin(h) / h);
            result[1] = complexType(-hy * std::sin(h) / h, -hx * std::sin(h) / h);
            result[2] = complexType(hy * std::sin(h) / h, -hx * std::sin(h) / h);
            result[3] = complexType(std::cos(h), hz * std::sin(h) / h);
        
            return result;
        }

        std::vector<complexType> UnitarySigmaISigmaJ(double Jx, double Jy, double Jz)
        {
            std::vector<complexType> result(16, complexType(0.0, 0.0));
        
            result[0] = std::exp(complexType(0.0, -Jz)) * complexType(std::cos(Jx - Jy), 0.0);
            result[3] = complexType(std::sin(Jx - Jy), 0.0) * complexType(-std::sin(Jz), -std::cos(Jz));
            result[5] = std::exp(complexType(0.0, Jz)) * complexType(std::cos(Jx + Jy), 0.0);
            result[6] = complexType(std::sin(Jx + Jy), 0.0) * complexType(std::sin(Jz), -std::cos(Jz));
            result[9]  = result[6];
            result[10] = result[5];
            result[12] = result[3];
            result[15] = result[0];
        
            return result;
        }

        std::vector<complexType> SigmaISigmaJSum(double Jx, double Jy, double Jz)
        {
            std::vector<complexType> result(16, complexType(0.0, 0.0));
        
            result[0] = complexType(Jz, 0.0);
            result[3] = complexType(Jx - Jy, 0.0);
            result[5] = complexType(-Jz, 0.0);
            result[6] = complexType(Jx + Jy, 0.0);
            result[9]  = result[6];
            result[10] = result[5];
            result[12] = result[3];
            result[15] = result[0];
        
            return result;
        }
    }
}