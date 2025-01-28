# Electrical Impedance Tomography (EIT) Solver

This repository contains a project for solving Electrical Impedance Tomography (EIT) problems. It includes a Windows Forms application for mesh generation, visualization, and solving forward problems using various numerical methods. The project is implemented in C# and uses libraries such as Math.NET and Triangle.NET for numerical and meshing tasks.

## Features
- Mesh generation: Supports rectangular and circular meshes.
- Solvers: Includes LU Decomposition, Singular Value Decomposition (SVD), and Conjugate Gradient (CG) methods for solving the forward problem.
- Visualization: Provides an interactive visualization of the generated mesh.
- Denoising: Supports multiple denoising techniques such as Gaussian, Median, and Kalman filters.
- Simulation options: Allows simulating symmetric, asymmetric, and noisy boundary measurements.

## Prerequisites
- .NET Framework 6.0 or higher
- IDE: Visual Studio or any compatible IDE for .NET development

## Setup Instructions
1. Clone the repository
2. Open the solution in Visual Studio or your preferred IDE.
3. Ensure the following libraries are installed via NuGet:
- `MathNet.Numerics`
- `Triangle.NET`
4. **Important**: Set the output type of the Windows Forms project to **Console Application** to enable console logging:
- Right-click on the project in the Solution Explorer.
- Select **Properties**.
- Under the **Application** tab, set the **Output type** to **Console Application**.
5. Build the project to restore dependencies and compile the code.

## How to Use
1. Run the application to launch the main interface.
2. Generate a mesh:
- Select either **Rectangular Mesh** or **Circular Mesh** and set the density (working with 0.5, 1, 2, 4, 8) in the input field.
3. Choose simulation parameters:
- Select a simulation type (e.g., symmetric, asymmetric, or noisy boundary conditions).
- Specify the regularization parameter if required by the solver.
4. Select a solver:
- Choose from LU Decomposition, SVD, or Conjugate Gradient methods in the dropdown menu.
5. Click **Solve** to compute the forward problem.
6. Visualize results directly in the application window.
7. Additional numerical details and logs will be displayed in the console.

## Code Overview
- `Mesh.cs`: Implements mesh generation and element handling.
- `Vertex.cs`: Defines vertices and their properties (e.g., boundary, potential).
- `Edge.cs`: Defines edges in the mesh, and their properties.
- `Element.cs`: Represents mesh elements, including shape function gradients and conductivity.
- `ForwardSolver.cs`: Handles the forward problem and stiffness matrix calculations.
- `DiscreteSolver.cs`: Implements numerical methods such as LU, SVD, and CG solvers.
- `Measurement.cs`: Contains methods for simulating and denoising boundary measurements.
- `MainForm.cs`: Windows Forms interface for user interaction.
- `Program.cs`: Entry point of the application.

## Contributions
Contributions are welcome! If you would like to contribute, please submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Known Errors
Mesh generation is kind of messed up if not using powers of 2. Assuming strong Dirichlet data (solving withour Lagrange multipliers) solves the system correctly, meaning the stiffness matrix is correctly assembled, and the produced potential map is consistent with other numerical solvers. When using lagrange multipliers (e.g., solving with LU, SVD, CG) the block saddle-point system is solved, and the reuslting potential distributions is quiet bad.

## Currently working on:
GMRES method to solve the block saddle-point system (maybe that helps stability with Lagrange multipliers), measurement structures, and patterns and the inverse solver using the adjoint state method approach.
