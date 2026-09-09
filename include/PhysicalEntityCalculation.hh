#pragma once

#include <vector>
#include <complex>

#include <cusolverDn.h>

#include "CuErrorUtils.hh"

namespace QTensorNet
{
    namespace PhysicalEntityCalculation
    {
        using complexType = std::complex<double>;

        double VonNeumannEntropy(const void* rho_matrix, size_t matrix_size);
    }
}