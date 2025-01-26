// EITInverseSolver.cs
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace EIT_SOLVER
{
    public class EITInverseSolver
    {
        public Mesh Mesh { get; set; }
        public double[] ObservedData { get; set; } // d_observed
        public double RegularizationParam { get; set; }
        public Vector<double> Sigma { get; private set; } // Current conductivity
        public Vector<double> Gradient { get; private set; }

        // Forward Solver
        private ForwardSolver forwardSolver;

        public EITInverseSolver(Mesh mesh, double[] observedData, double regularizationParam)
        {
            Mesh = mesh;
            ObservedData = observedData;
            RegularizationParam = regularizationParam;
            InitializeConductivity();
            //forwardSolver = new ForwardSolver(mesh, observedData);
        }

        private void InitializeConductivity()
        {
            int N_elements = Mesh.Elements.Count;
            Sigma = Vector<double>.Build.Dense(N_elements, 1.0); // Initial guess: σ = 1 everywhere
        }

        public void Iterate(int maxIterations, double tolerance)
        {
            for (int k = 0; k < maxIterations; k++)
            {
                // Step 1: Solve Forward Problem with current Sigma
                //forwardSolver.Solve();

                // Step 2: Compute Misfit and Gradient
                ComputeGradient();

                // Step 3: Update Sigma
                double stepSize = 0.01; // Can be optimized
                Sigma = Sigma - stepSize * Gradient;

                // Step 4: Check Convergence
                if (Gradient.L2Norm() < tolerance)
                    break;
            }
        }

        private void ComputeGradient()
        {
            // Assuming forwardSolver has Alpha and Gamma vectors
            // Gradient: ∇λ ⋅ ∇φ
            // Simplified computation: Element-wise product
            Gradient = Vector<double>.Build.Dense(Sigma.Count, 0.0);
            for (int e = 0; e < Mesh.Elements.Count; e++)
            {
                // Compute gradients
                var element = Mesh.Elements[e];
                double gradPhiDotGradLambda = 0.0;
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        // Local gradients
                        double gradPhi_i = element.GradPhi[i, 0];
                        double gradPhi_j = element.GradPhi[i, 1];
                        double gradLambda_i = element.GradPhi[j, 0]; // Placeholder
                        double gradLambda_j = element.GradPhi[j, 1]; // Placeholder
                        gradPhiDotGradLambda += gradLambda_i * gradPhi_i + gradLambda_j * gradPhi_j;
                    }
                }
                Gradient[e] = gradPhiDotGradLambda;
            }

            // Add regularization gradient (e.g., Tikhonov)
            for (int e = 0; e < Sigma.Count; e++)
            {
                Gradient[e] += RegularizationParam * (Sigma[e] - 1.0); // Assuming σ_prior = 1
            }
        }
    }
}
