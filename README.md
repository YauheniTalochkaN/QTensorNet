![Logo](./logo.png)

# QTensorNet
This repository contains a cuQuantum-based toolkit designed for many-body problem simulations using tensor networks (TNs).

# Examples
- **`TEBD1D`** implements the TEBD algorithm using the second-order Suzuki–Trotter decomposition to simulate the dynamics of a local 1D spin chain.
- **`RK1D`** implements the fourth-order Runge–Kutta method using the summation and vector-operator product operations defined for two TNs to simulate the dynamics of a non-local 1D spin network.
- **`DMRGSquareKagome`** and **`DMRGHoneycomb`** apply the DMRG algorithm implemented in QTensorNet to evaluate the eigenvalue problem of non-local 2D spin networks on decorated square-kagome and honeycomb lattices, respectively.
- **`TDVP3DSchrodinger`** and **`TDVP3DvonNeumann`** apply the TDVP algorithm implemented in QTensorNet to simulate the dynamics of a non-local 3D spin network using the Schrödinger and von Neumann equations, respectively.

# License
This project is licensed under the [CC BY-NC-ND 4.0 license](LICENSE.md).