#define _LOGGING_

using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Runtime.InteropServices;
using Microsoft.VisualBasic.Logging;
using static System.Windows.Forms.Design.AxImporter;

namespace EIT_SOLVER
{
    static public class DiscreteSolver
    {
        // Solve System with LU decomposition
        private static void SolveSystemLU(int N_phi, int N_lambda, double[,] A, double[] b, ref double[] Alpha, ref double[] Gamma, double Theta)
        {
            // Create a big Matrix<double> of size (N_phi + N_lambda) x (N_phi + N_lambda).

            int size = N_phi + N_lambda;
            var bigA = DenseMatrix.Create(size, size, 0.0);
            var rhsVec = DenseVector.Create(size, 0.0);

            bigA = bigA + Theta * DenseMatrix.CreateIdentity(size);

            // Now we have: bigA * [ alpha; gamma ] = rhsVec

            // Solve the system
            var lu = bigA.LU(); // or .Svd(), .Qr(), etc
            var solution = lu.Solve(rhsVec);

            // Get the coefficients
            for (int i = 0; i < N_phi; i++)
            {
                Alpha[i] = solution[i];
            }

            for (int j = 0; j < N_lambda; j++)
            {
                Gamma[j] = solution[N_phi + j];
            }
#if _LOGGING_
            Console.WriteLine("Solution alpha[]:");
            for (int i = 0; i < N_phi; i++)
            {
                Console.WriteLine($"  alpha[{i}] = {Alpha[i]}");
            }
            Console.WriteLine("Solution gamma[]:");
            for (int j = 0; j < N_lambda; j++)
            {
                Console.WriteLine($"  gamma[{j}] = {Gamma[j]}");
            }
#endif
        }

        // Solve System with SVD
        private static void SolveSystemSVD(int N_phi, int N_lambda, double[,] A, double[] b, ref double[] Alpha, ref double[] Gamma, double Theta)
        {
            // Create a big Matrix<double> of size (N_phi + N_lambda) x (N_phi + N_lambda).
            int size = N_phi + N_lambda;
            var bigA = DenseMatrix.Create(size, size, 0.0);
            var rhsVec = DenseVector.Create(size, 0.0);

            // Solve the system using SVD
            Vector<double> solution;
            try
            {
                // Perform Singular Value Decomposition
                var svd = bigA.Svd(true); // Compute thin SVD

                // Check the condition number to assess matrix invertibility
                double conditionNumber = svd.ConditionNumber;
                if (double.IsInfinity(conditionNumber) || double.IsNaN(conditionNumber))
                {
                    MessageBox.Show("The system matrix is singular or ill-conditioned and cannot be solved reliably.", "SVD Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                // Solve the linear system using SVD
                solution = svd.Solve(rhsVec);

                if (solution == null)
                {
                    MessageBox.Show("SVD failed to find a solution.", "SVD Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred during SVD-based solving: {ex.Message}", "SVD Exception", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // Extract the coefficients into Alpha and Gamma
            for (int i = 0; i < N_phi; i++)
                Alpha[i] = solution[i];

            for (int j = 0; j < N_lambda; j++)
                Gamma[j] = solution[N_phi + j];

#if _LOGGING_
            Console.WriteLine("Solution alpha[]:");
            for (int i = 0; i < N_phi; i++)
                Console.WriteLine($"  alpha[{i}] = {Alpha[i]}");
            Console.WriteLine("Solution gamma[]:");
            for (int j = 0; j < N_lambda; j++)
                Console.WriteLine($"  gamma[{j}] = {Gamma[j]}");
#endif
        }

        public static Vector<double> SolveTruncatedSVD(double[,] A, double[] b, double svdCutoff)
        {
            //int m = A.RowCount;
            //int n = A.ColumnCount;
            //if (b.Count != m)
            //    throw new ArgumentException("Dimensions of A and b do not match.");

            // 1) Compute the full SVD of A. 
            //    Math.NET returns:
            //      A = U * S * VT 
            //    where 
            //      U is (m x r),
            //      S is a diagonal (r x r) but stored as vector,
            //      VT is (r x n),
            //    and r = min(m,n).
            //var svd = A.Svd(computeVectors: true);  // full SVD

            //// Extract pieces
            //Vector<double> singularValues = svd.S;   // length r
            //Matrix<double> U = svd.U;               // (m x r)
            //Matrix<double> VT = svd.VT;             // (r x n)

            //// 2) We will build the solution xTrunc in R^n
            //var xTrunc = Vector<double>.Build.Dense(n, 0.0);

            //// 3) Compute c = U^T * b  (an r-dimensional vector)
            ////    Because U is (m x r), U^T is (r x m).
            //Vector<double> UTb = U.TransposeThisAndMultiply(b);

            //// 4) Loop over each singular value s_i
            ////    If s_i > svdCutoff, accumulate ( (c_i / s_i) * v_i ) into xTrunc
            //int r = singularValues.Count;  // r = rank or min(m,n)
            //for (int i = 0; i < r; i++)
            //{
            //    double s = singularValues[i];
            //    if (Math.Abs(s) > svdCutoff)
            //    {
            //        // scale = c_i / s_i
            //        double scale = UTb[i] / s;

            //        // The i-th right singular vector is row i of V^T => (VT[i,:]) 
            //        // which has length n.
            //        // xTrunc += scale * (the row i of VT).
            //        for (int col = 0; col < n; col++)
            //        {
            //            xTrunc[col] += scale * VT[i, col];
            //        }
            //    }
            //    // else if s <= svdCutoff, we skip it => effectively discarding 
            //    // the small singular direction.
            //}

            // return xTrunc;
            Vector<double> ret = new DenseVector(1);

            return ret;
        }

        private static void SolveSystemCG(int N_phi, int N_lambda, Matrix K, Matrix B, double[] f, ref double[] Alpha, ref double[] Gamma, double Theta, bool precon = false)
        {
            // Ensure that K is symmetric positive definite
            if (!IsSymmetric(K))
            {
                MessageBox.Show("Matrix K is not symmetric.", "Solver Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            //if (!IsPositiveDefinite(K))
            //{
            //    MessageBox.Show("Matrix K is not positive definite.", "Solver Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            //    return;
            //}

            // Step 1: Compute Schur Complement S = B^T * K^{-1} * B + Theta * I
            // Since directly computing K^{-1} is expensive, we'll define a function to compute S * gamma

            // Define a lambda function for matrix-vector multiplication with S
            Func<Vector<double>, Vector<double>> S_Multiply = (Vector<double> gamma) =>
            {
                // Compute B * gamma
                Vector<double> B_gamma = B * gamma;

                // Solve K * x = B_gamma for x using Conjugate Gradient

                if(!precon)
                {
                    Vector<double>? x = ConjugateGradientSolve(K, B_gamma, 1e-10, 100);
                    if (x == null)
                        throw new Exception("Conjugate Gradient solver failed to solve K * x = B * gamma.");

                    // Compute B^T * x
                    Vector<double> B_T_x = B.Transpose() * x;

                    // Add Theta * gamma
                    Vector<double> result = B_T_x + (Theta * gamma);

                    return result;
                }
                else
                {
                    Vector<double>? x = ConjugateGradientSolve_Preconditioned(K, B_gamma, 1e-10, 100);
                    if (x == null)
                        throw new Exception("Conjugate Gradient solver failed to solve K * x = B * gamma.");

                    // Compute B^T * x
                    Vector<double> B_T_x = B.Transpose() * x;

                    // Add Theta * gamma
                    Vector<double> result = B_T_x + (Theta * gamma);

                    return result;
                }
            };

            // Step 2: Use Conjugate Gradient to solve S * gamma = f
            // Note: If your system is S * gamma = -f, adjust accordingly
            // Here, assuming S * gamma = f based on original LU and SVD implementations

            Vector<double> f_vec = Vector<double>.Build.Dense(f.Length, 0.0);
            for (int j = 0; j < N_lambda; j++)
                f_vec[j] = f[j];

            // Define the right-hand side for Schur Complement system
            Vector<double> rhs_S = f_vec.SubVector(0, N_lambda); // Assuming f corresponds to the second block

            // Implement the CG solver for S * gamma = rhs_S
            Vector<double>? gamma_solution = ConjugateGradientSolve_SchurComplement(S_Multiply, rhs_S, 1e-10, 100);

            if (gamma_solution == null)
            {
                MessageBox.Show("Conjugate Gradient solver failed to solve the Schur Complement system.", "Solver Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // Step 3: Solve for Alpha using gamma
            // Alpha = K^{-1} * B * gamma
            Vector<double> B_gamma_solution = B * gamma_solution;
            Vector<double>? alpha_solution = ConjugateGradientSolve(K, B_gamma_solution, 1e-10, 1000);

            if (alpha_solution == null)
            {
                MessageBox.Show("Conjugate Gradient solver failed to solve K * Alpha = B * gamma.", "Solver Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // Step 4: Assign solutions to Alpha and Gamma arrays
            for (int i = 0; i < N_phi; i++)
                Alpha[i] = alpha_solution[i];

            for (int j = 0; j < N_lambda; j++)
                Gamma[j] = gamma_solution[j];

#if _LOGGING_
            Console.WriteLine("Solution alpha[]:");
            for (int i = 0; i < N_phi; i++)
                Console.WriteLine($"  alpha[{i}] = {Alpha[i]}");

            Console.WriteLine("Solution gamma[]:");
            for (int j = 0; j < N_lambda; j++)
                Console.WriteLine($"  gamma[{j}] = {Gamma[j]}");
#endif
        }

        // Helper method to solve S * gamma = rhs using Conjugate Gradient
        private static Vector<double>? ConjugateGradientSolve_SchurComplement(Func<Vector<double>, Vector<double>> S_Multiply, Vector<double> rhs, double tolerance, int maxIterations)
        {
            Vector<double> x = Vector<double>.Build.Dense(rhs.Count, 0.0); // Initial guess
            Vector<double> r = rhs - S_Multiply(x);
            Vector<double> p = r.Clone();
            double rsold = r.DotProduct(r);

            for (int i = 0; i < maxIterations; i++)
            {
                Vector<double> Ap = S_Multiply(p);
                double alpha = rsold / p.DotProduct(Ap);
                x += alpha * p;
                r -= alpha * Ap;
                double rsnew = r.DotProduct(r);

                if (Math.Sqrt(rsnew) < tolerance)
                    break;

                p = r + (rsnew / rsold) * p;
                rsold = rsnew;
            }

            // Check for convergence
            if (Math.Sqrt(rsold) >= tolerance)
                return null; // Convergence not achieved

            return x;
        }

        // Helper method to solve K * x = b using Conjugate Gradient
        private static Vector<double>? ConjugateGradientSolve(Matrix<double> K, Vector<double> b, double tolerance, int maxIterations)
        {
            // Ensure K is symmetric
            if (!IsSymmetric(K))
            {
                MessageBox.Show("Matrix K is not symmetric.", "Solver Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return null;
            }

            // Initialize variables
            Vector<double> x = Vector<double>.Build.Dense(K.ColumnCount, 0.0); // Initial guess
            Vector<double> r = b - (K * x);
            Vector<double> p = r.Clone();
            double rsold = r.DotProduct(r);

            for (int i = 0; i < maxIterations; i++)
            {
                Vector<double> Ap = K * p;
                double alpha = rsold / p.DotProduct(Ap);
                x += alpha * p;
                r -= alpha * Ap;
                double rsnew = r.DotProduct(r);

                if (Math.Sqrt(rsnew) < tolerance)
                    break;

                p = r + (rsnew / rsold) * p;
                rsold = rsnew;
            }

            // Check for convergence
            if (Math.Sqrt(rsold) >= tolerance)
                return null;   // Convergence not achieved

            return x;
        }

        // Helper method to check if a matrix is symmetric
        private static bool IsSymmetric(Matrix<double> matrix, double tolerance = 1e-6)
        {
            if (matrix.RowCount != matrix.ColumnCount)
                return false;

            for (int i = 0; i < matrix.RowCount; i++)
                for (int j = i + 1; j < matrix.ColumnCount; j++)
                    if (Math.Abs(matrix[i, j] - matrix[j, i]) > tolerance)
                        return false;

            return true;
        }

        // Helper method to check if a matrix is positive definite
        private static bool IsPositiveDefinite(Matrix<double> matrix)
        {
            try
            {
                var cholesky = matrix.Cholesky();
                return true;
            }
            catch
            {
                return false;
            }
        }

        // Helper method to create a Jacobi preconditioner
        private static Vector<double> CreateJacobiPreconditioner(Matrix<double> K)
        {
            Vector<double> preconditioner = Vector<double>.Build.Dense(K.ColumnCount, 0.0);
            for (int i = 0; i < K.RowCount; i++)
                preconditioner[i] = 1.0 / K[i, i];
            return preconditioner;
        }

        // Modified CG solver with Jacobi preconditioning
        private static Vector<double>? ConjugateGradientSolve_Preconditioned(Matrix<double> K, Vector<double> b, double tolerance, int maxIterations)
        {
            // Ensure K is symmetric
            if (!IsSymmetric(K))
            {
                MessageBox.Show("Matrix K is not symmetric.", "Solver Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return null;
            }

            // Initialize variables
            Vector<double> x = Vector<double>.Build.Dense(K.ColumnCount, 0.0); // Initial guess
            Vector<double> r = b - (K * x);
            Vector<double> z = r.PointwiseMultiply(CreateJacobiPreconditioner(K));
            Vector<double> p = z.Clone();
            double rsold = r.DotProduct(z);

            for (int i = 0; i < maxIterations; i++)
            {
                Vector<double> Ap = K * p;
                double alpha = rsold / p.DotProduct(Ap);
                x += alpha * p;
                r -= alpha * Ap;
                Vector<double> z_new = r.PointwiseMultiply(CreateJacobiPreconditioner(K));
                double rsnew = r.DotProduct(z_new);

                if (Math.Sqrt(rsnew) < tolerance)
                    break;

                p = z_new + (rsnew / rsold) * p;
                rsold = rsnew;
            }

            // Check for convergence
            if (Math.Sqrt(rsold) >= tolerance)
                return null;// Convergence not achieved

            return x;
        }

        // Solve the system assuming strong Dirichlet data
        private static void SolveSystemWithoutLagrangeMultipliers(int N_phi, int N_lambda, double[,] K, double[,] B, double[] f, ref double[] Alpha, ref double[] Gamma, double Theta, double[] BoundaryVoltages)
        {
            f = new double[N_phi];

            for (int i = 0; i < N_phi; i++)
                f[i] = BoundaryVoltages[i];

            var bigA = DenseMatrix.Create(N_phi, N_phi, 0.0);
            var rhsVec = DenseVector.Create(N_phi, 0.0);

            for (int i = 0; i < N_phi; i++)
                for (int j = 0; j < N_phi; j++)
                    bigA[i, j] = K[i, j];

            for (int i = 0; i < N_phi; i++)
                rhsVec[i] = f[i];

            var lu = bigA.LU();
            var solution = lu.Solve(rhsVec);

            // Get the coefficients
            for (int i = 0; i < N_phi; i++)
                Alpha[i] = solution[i];

#if _LOGGING_
            Console.WriteLine("Solution alpha[]:");
            for (int i = 0; i < N_phi; i++)
                Console.WriteLine($"  alpha[{i}] = {Alpha[i]}");
#endif
        }

        public static Vector<double> SolveGmres(Matrix<double> A, Vector<double> b, Vector<double> x0, int restart = 50, int maxIterations = 100, double tolerance = 1e-8)
        {
            // Initial guess
            x0 = Vector<double>.Build.Dense(b.Count, 0.0);
            var retVal = GmresSolver.Solve(A, b, maxIterations, 100, tolerance, x0, degreeOfParallelism: 4);

            Console.WriteLine($"Converged: {retVal.IsConverged}.");

            return retVal.X;
        }


        public static void Solve(string SolverType, int N_phi, int N_lambda, double[,] K, double[,] B, double[] f, double[,] A, double[] b, ref double[] Alpha, ref double[] Gamma, double Theta, double[] BoundaryVoltages)
        {
            if (SolverType == "LU" || SolverType == "lu" || SolverType == "Lu" || SolverType == "lU")
                SolveSystemLU(N_phi, N_lambda,A, b, ref Alpha, ref Gamma, Theta);
            else if (SolverType == "SVD" || SolverType == "svd")
                SolveSystemSVD(N_phi, N_lambda, A, b, ref Alpha, ref Gamma, Theta);
            else if(SolverType == "TSVD" || SolverType == "TruncatedSVD")
            {
                Vector<double> sol = SolveTruncatedSVD(A, b, 1e-8);
                for (int i = 0; i < sol.Count; i++)
                {
                    if (i < N_phi)
                        Alpha[i] = sol[i];
                    else
                        Gamma[i - N_phi] = sol[i];
                }

            }
            else if (SolverType == "CG" || SolverType == "ConjugateGradient" || SolverType == "Conjugate Gradient")
            {
                double[] tmp = new double[N_phi * N_phi];
                for (int i = 0; i < N_phi; i++)
                    for (int j = 0; j < N_phi; j++)
                        tmp[i + i * j] = K[i, j];

                double[] tmp2 = new double[N_phi * N_lambda];
                for (int i = 0; i < N_phi; i++)
                    for (int j = 0; j < N_lambda; j++)
                        tmp2[i + i * j] = B[i, j];


                Matrix k = new DenseMatrix(N_phi, N_phi, tmp);
                Matrix bb = new DenseMatrix(N_phi, N_lambda, tmp2);
                SolveSystemCG(N_phi, N_lambda, k, bb, f, ref Alpha, ref Gamma, Theta);

            }
            else if (SolverType == "CGprecon" || SolverType == "CGpreconditioned")
            {
                //SolveSystemCG(N_phi, N_lambda, K, B, f, ref Alpha, ref Gamma, Theta, true);
            }
            else if (SolverType == "WithoutLagrange" || SolverType == "WithoutLagrangeMultipliers" || SolverType == "withourlagrange" || SolverType == "w.o.l.")
                SolveSystemWithoutLagrangeMultipliers(N_phi, N_lambda, K, B, f, ref Alpha, ref Gamma, Theta, BoundaryVoltages);
            else if (SolverType == "GMRES" || SolverType == "GreatestMinimumResidual")
            {
                //Matrix a = new DenseMatrix(N_phi + N_lambda, N_phi + N_lambda, A);
                var bigA = DenseMatrix.Create(N_phi + N_lambda, N_phi + N_lambda, 0.0);
                var rhsVec = DenseVector.Create(N_phi + N_lambda, 0.0);

                for (int i = 0; i < N_phi; i++)
                    for (int j = 0; j < N_phi; j++)
                        bigA[i, j] = K[i, j];

                // Fill top-right block with B
                // Fill bottom-left block with B^T
                for (int i = 0; i < N_phi; i++)
                {
                    for (int j = 0; j < N_lambda; j++)
                    {
                        bigA[i, N_phi + j] = B[i, j];
                        bigA[N_phi + j, i] = B[i, j];
                    }
                }

                for (int i = 0; i < N_lambda; i++)
                    rhsVec[N_phi + i] = f[i];

                Vector<double> x0 = DenseVector.Create(N_phi + N_lambda, 0.0);

                Vector<double> sol = SolveGmres(bigA, rhsVec, x0, maxIterations: 100, tolerance: 1e-8);

                for (int i = 0; i < sol.Count; i++)
                {
                    if (i < N_phi)
                        Alpha[i] = sol[i];
                    else
                        Gamma[i - N_phi] = sol[i];
                }
            }
            else
            {
                throw new ArgumentOutOfRangeException(nameof(SolverType), "Invalid solver type specified.");
            }

#if _LOGGING_
            Console.WriteLine("Logging output vectors from calculations.");
            LogOutputVectors(N_phi, N_lambda, Alpha, Gamma);
#endif
        }

        private static void LogOutputVectors(int N_phi, int N_lambda, double[] Alpha, double[] Gamma)
        {
#if _LOGGING_
            Console.WriteLine("Solution alpha[]:");
            for (int i = 0; i < N_phi; i++)
                Console.WriteLine($"  alpha[{i}] = {Alpha[i]}");
            Console.WriteLine("Solution gamma[]:");
            for (int j = 0; j < N_lambda; j++)
                Console.WriteLine($"  gamma[{j}] = {Gamma[j]}");
#endif
        }
    }
}
