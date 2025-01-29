// EITInverseSolver.cs
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NUnit.Framework.Constraints;
using TriangleNet;
using TriangleNet.Meshing;

namespace EIT_SOLVER
{
    public class InverseSolver
    {
        private Mesh Mesh { get; set; }
        private double[] ObservedData { get; set; } // d_observed
        private int MaxIterations { get; set; }  // Maximum iteration count
        private double Tolerance { get; set; }  // Tolerance or stopping threshold
        public double Theta { get; set; }   // Regularizatuin parameter
        public double Beta { get; set; } // Learning rate
        public double[] Sigma { get; private set; } // Current conductivity
        private ForwardSolver ForwardSolver { get; set; }  // Forward Solver

        public InverseSolver(Mesh mesh,
                             double[] initialSigma,
                             double[] observedData,
                             double beta = 1e-3,
                             int maxIterations = 50,
                             double tolerance = 1e-6,
                             double theta= 0.0)
        {
            Mesh = mesh;
            Sigma = initialSigma;  // user-provided initial guess
            ObservedData= observedData;
            Beta = beta;
            MaxIterations = maxIterations;
            Tolerance = tolerance;
            Theta = theta;

            // Initialize a ForwardSolver that uses this mesh & boundary voltages
            // (We will overwrite the mesh's element sigma values from 'Sigma'.)
            ForwardSolver = new ForwardSolver(mesh, observedData);
        }

        private void InitializeConductivity()
        {
            int N_elements = Mesh.Elements.Count;
            //Sigma = Vector<double>.Build.Dense(N_elements, 1.0); // Initial guess: σ = 1 everywhere
        }

        // Perform the iterative reconstruction of the conductivity sigma.
        public void SolveInverse(string solverType)
        {
            int iteration = 0;
            double residualNorm = double.PositiveInfinity;

            for (iteration = 0; iteration < MaxIterations; iteration++)
            {
                // Set the mesh element conductivities to Sigma:
                SetMeshConductivities(Sigma);

                // Solve the forward problem to get 'Alpha' (the interior potential dofs).
                //    We'll do: forwardSolver.Solve("LU Decomposition") or your solver choice.
                ForwardSolver.Solve(solverType);

                // The forward solution is now in forwardSolver.Alpha (internal nodes).
                // The boundary potential is in forwardSolver.BoundaryVoltages, 
                // which is set to 'observedBoundaryVoltages' in this example.

                // Solve the Adjoint problem to get 'Gamma'.
                //    For EIT with a misfit = 1/2||phi_boundary - data||^2,
                //    the adjoint PDE basically has boundary condition = (phi_boundary - data).
                //    One simple approach is:
                double[] adjointBoundary = ComputeAdjointBoundary(ForwardSolver.Alpha, ObservedData);

                // Make a new ForwardSolver or re-use with special boundary voltages for the adjoint:
                ForwardSolver adjointSolver = new ForwardSolver(Mesh, adjointBoundary);
                // Typically the PDE is the same: grad·(sigma grad lam) = [something].
                // For the simplest approach, if the PDE is the same form, the "f" part 
                // changes to reflect (phi_boundary - data).
                adjointSolver.Solve(solverType);

                // Now the adjoint solution is in adjointSolver.Alpha (the interior dofs).
                // We'll interpret that as 'Gamma' in your notation.
                // We do NOT care about the boundary dofs in the same sense; the interior dofs 
                // represent the adjoint variable in the domain.

                // 4) Compute the gradient w.r.t. sigma. 
                //    We use eqn: grad_sigma ~ \sum_{element} (nabla phi · nabla lambda)*area
                //    We'll do it element by element:
                double[] gradient = ComputeGradient(Mesh,
                                                   Sigma,
                                                   ForwardSolver.Alpha,      // "phi"
                                                   adjointSolver.Alpha);     // "lambda"

                // 5) (Optionally) Add Tikhonov or other regularization gradient:
                if (Theta > 0.0)
                {
                    // Zero-order Tikhonov example:  grad_reg = regParameter * (Sigma - SigmaPrior).
                    // Suppose we have no "SigmaPrior" -> use 0 for simplicity:
                    for (int e = 0; e < gradient.Length; e++)
                    {
                        gradient[e] += Theta * Sigma[e];
                    }
                }

                // 6) Update sigma:
                double[] newSigma = new double[Sigma.Length];
                for (int e = 0; e < Sigma.Length; e++)
                {
                    newSigma[e] = Sigma[e] - Beta * gradient[e];
                }

                // 7) Compute a "residual" or "step norm" to see if we are done:
                residualNorm = 0.0;
                for (int e = 0; e < Sigma.Length; e++)
                {
                    double diff = newSigma[e] - Sigma[e];
                    residualNorm += diff * diff;
                }
                residualNorm = Math.Sqrt(residualNorm);

                // Overwrite Sigma with newSigma
                Sigma = newSigma;

                Console.WriteLine($"Iter={iteration}, Step Norm={residualNorm:F6}");

                // 8) Check convergence
                if (residualNorm < Tolerance)
                {
                    break;
                }
            }

            Console.WriteLine($"Inverse solve finished after {iteration} iterations.  Final stepNorm={residualNorm}");
        }

        /// <summary>
        /// Compute the boundary condition for the adjoint problem: 
        ///    lam_boundary = (phi_forward_boundary - observedData).
        /// For a least-squares misfit 1/2 * ||phi_boundary - data||^2, 
        /// the natural approach is lam_boundary = (phi_boundary - data).
        /// </summary>
        private double[] ComputeAdjointBoundary(double[] alphaForward, double[] dataObserved)
        {
            // alphaForward typically is only the interior dofs, 
            // but let's assume your forwardSolver stores 
            // boundary pot. in forwardSolver.BoundaryVoltages anyway. 
            // So let's get them from forwardSolver.BoundaryVertices or from the mesh.

            // But in this snippet, let's assume dataObserved has length == boundaryNodeCount
            // and forwardSolver also has boundary voltages the same length.
            // So the adjoint boundary condition is simply (phiBoundary - data).
            // We'll do it element-wise:
            double[] lamBoundary = new double[dataObserved.Length];



            // Suppose forwardSolver's boundary voltages are simply 'observedBoundaryVoltages' 
            // or we've stored the actual forward boundary in some array. 
            // Let that be 'phiBoundary[i]'. We'll get it from mesh.BoundaryVertices[i].Potential
            for (int i = 0; i < lamBoundary.Length; i++)
            {
                double phi_i = Mesh.BoundaryVertices[i].Potential;   // forward boundary solution
                double data_i = dataObserved[i];
                lamBoundary[i] = phi_i - data_i;
            }

            return lamBoundary;
        }

        /// <summary>
        /// Compute gradient wrt sigma using: 
        ///    dJ/dsigma ~ \int_{elem} (nabla phi · nabla lambda) dx
        /// for each element, plus the factor of element area (and possibly the local sigma if your formula uses that).
        /// 
        /// We will access the per-element shape function gradients, 
        /// and the dof values from alphaForward, alphaAdjoint.
        /// </summary>
        private double[] ComputeGradient(Mesh mesh,
                                         double[] currentSigma,
                                         double[] alphaForward,   // interior dofs for phi
                                         double[] alphaAdjoint)   // interior dofs for lambda
        {
            double[] grad = new double[mesh.Elements.Count];

            // The mesh presumably stores each element's "DotProducts" of shape fn gradients, 
            // or you can reconstruct from e.GradPhi. 
            // Then you combine alphaForward's relevant dofs with alphaAdjoint's relevant dofs.
            // 
            // We'll do a simple approach: 
            //   For each element e, 
            //     reconstruct "grad phi" and "grad lambda" at that element (constant in linear FE),
            //     dot them, multiply by area, store in grad[e].
            // 
            // Actually building "grad phi" from alphaForward requires summing alpha_i * d phi_i/dx 
            // for i in the local vertices. 
            // Similarly for lambda. Then dot them. 
            // 
            // This code is only a rough outline. You’ll adapt to your data structures.

            for (int eIdx = 0; eIdx < mesh.Elements.Count; eIdx++)
            {
                Element elem = mesh.Elements[eIdx];

                // We'll fetch the local vertex indices for the interior nodes.
                Vertex[] verts = { elem.V1, elem.V2, elem.V3 };

                // Build gradPhi in R^2
                double gradPhiX = 0.0;
                double gradPhiY = 0.0;

                double gradLamX = 0.0;
                double gradLamY = 0.0;

                for (int loc = 0; loc < 3; loc++)
                {
                    // If vertex is interior, domainIndex >= 0 => alphaForward index
                    // If vertex is boundary, domainIndex < 0 => skip or is 0
                    int index = verts[loc].DomainIndex;
                    if (index >= 0)
                    {
                        double aF = alphaForward[index]; // alpha_i
                        // d phi_i/dx in element => elem.GradPhi[loc,0], etc. 
                        // or from elem.DotProducts if you stored them differently
                        // Suppose elem.GradPhi is shape (3,2): 
                        //   GradPhi[loc,0] = d phi_loc / dx
                        //   GradPhi[loc,1] = d phi_loc / dy
                        gradPhiX += aF * elem.GradPhi[loc, 0];
                        gradPhiY += aF * elem.GradPhi[loc, 1];

                        double aLam = alphaAdjoint[index]; // gamma_i
                        gradLamX += aLam * elem.GradPhi[loc, 0];
                        gradLamY += aLam * elem.GradPhi[loc, 1];
                    }
                }

                // Now dot them:
                double dotVal = gradPhiX * gradLamX + gradPhiY * gradLamY;

                // Multiply by element area (which is stored in elem.Area).
                grad[eIdx] = dotVal * elem.Area;

                // If the formula is exactly ∂J/∂σ = ∫ (nabla phi · nabla lam), 
                // we do not multiply by the local sigma. 
                // But in some derivations, you see "sigma * (..)" if the PDE is set differently. 
                // Check your final expression carefully.
                // Optionally: grad[eIdx] *= currentSigma[eIdx];

            }

            return grad;
        }

        private void SetMeshConductivities(double[] sigma)
        {
            for (int i = 0; i < Mesh.Elements.Count; i++)
            {
                Mesh.Elements[i].Sigma = sigma[i];
            }
        }
    }
}
