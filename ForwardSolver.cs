// ForwardSolver.cs 
//#define _LOGGING_

using EIT_SOLVER;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Reflection.Metadata.Ecma335;
using System.Windows.Forms.VisualStyles;
using System.Xml.Linq;
using TriangleNet;
using TriangleNet.Geometry;
using static EIT_SOLVER.Measurement;
namespace EIT_SOLVER
{
    public class ForwardSolver
    {
        public Mesh Mesh { get; }   // Mesh object holds all data for finite-element discretization
        public double[] BoundaryVoltages { get; }   // Measured voltages on the boundary
        private double[,] K { get; set; }   // Stiffness matrix
        private double[,] B { get; set; }   // Boundary matrix
        private double[] f { get; set; }    // Load vector
        public double[] Alpha; // Coefficients for φ
        private double[] Gamma; // Coefficients for λ
        private int N_phi { get; set; } = 0;    // Number of φ coefficients
        private int N_lambda { get; set; } = 0; // Number of λ coefficients
        public double Theta { get; set; } = 1.0; // Regularization parameter
        public Measurement Measurement;
        public DenoisingMethod DenoisingMethod;

        public ForwardSolver(Mesh mesh, double[] boundaryVoltages)
        {
            Mesh = mesh;
            BoundaryVoltages = boundaryVoltages;

            // Global counts of unknowns
            N_phi = Mesh.InternalVertices.Count();
            N_lambda = Mesh.BoundaryVertices.Count();

            if (N_phi <= 0 || N_lambda <= 0)
                throw new ArgumentException(nameof(mesh), "N_phi or N_lambda was 0 during initialization, check mesh generation errors!");

            // Creating matrices and vectors
            K = new double[N_phi, N_phi];
            B = new double[N_phi, N_lambda];
            f = new double[N_lambda];
            Alpha = new double[N_phi];
            Gamma = new double[N_lambda];

            Measurement = new Measurement(N_lambda);
        }

        private double[,] CalculateElementStiffnessMatrix(Element element)
        {
            // Get the vertexes of the given element
            Vertex vertex1 = element.V1;
            Vertex vertex2 = element.V2;
            Vertex vertex3 = element.V3;

            // Get the area of the element
            double area = element.Area;

            // Get the gradient vectors from the given element
            double[,] gradients = element.GradPhi;

            // Get the conductivity of the given element
            double sigma = element.Sigma;

            // Create local stiffness matrix, that will store tha values
            double[,] localStiffnessMatrix = new double[3, 3];

            // Store the dotproduct of the gradients
            double[,] dotProducts = element.DotProducts;

            for (int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    // K_ij = \sigma_T |T| [\grad(\phi_i^T \cdot \grad(\phi_j^T))]
                    localStiffnessMatrix[i, j] = sigma * area * dotProducts[i, j];
                }
            }

#if _LOGGING_
            Console.Write("[");
            for(int i = 0; i < 3;i++)
            {
                for (int j = 0; j < 3; j++)
                    Console.Write("{0} \t", localStiffnessMatrix[i, j]);
                Console.WriteLine();
            }
            Console.WriteLine("]");
#endif

            return localStiffnessMatrix;
        }

        private void CalculateStiffnessMatrix()
        {
            // Clear the stiffness matrix
            K = new double[N_phi, N_phi];

            // Get each local stiffness matrix and add it to the global stiffness matrix
            foreach (Element element in Mesh.Elements)
            {
                int rowIndex = 0;
                int colIndex = 0;

                // Calculate the local stiffness matrix
                double[,] localStiffnessMatrix = CalculateElementStiffnessMatrix(element);

                // Retrieve the vertices of the element
                Vertex[] elementVertices = { element.V1, element.V2, element.V3 };

                // Add the local stiffness matrix to the global stiffness matrix
                for (int i = 0; i < 3; i++)
                {
                    // Row index is the global index of the i-th vertex
                    rowIndex = elementVertices[i].DomainIndex;

                    // Skip boundary vertices
                    if (rowIndex < 0) continue;

                    for (int j = 0; j < 3; j++)
                    {
                        // Column index is the global index of the j-th vertex
                        colIndex = elementVertices[j].DomainIndex;

                        // Skip boundary vertices
                        if (colIndex < 0) continue;

                        K[rowIndex, colIndex] += localStiffnessMatrix[i, j];
                    }
                }
            }

#if _LOGGING_
            Console.WriteLine("The stiffness matrix:");
            for (int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_phi; j++)
                {
                    if (K[i, j] == 0)
                        Console.Write(" " + K[i, j].ToString("F2") + " "); //Console.Write("0 ");// 
                    else
                        Console.Write(K[i, j].ToString("F2") + " ");//Console.Write("1 ");//

                }
                Console.Write(";");
                Console.WriteLine();
            }
            int nonZero = 0;
            for (int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_phi; j++)
                    if (K[i, j] != 0.0)
                        nonZero++;
            }

            Console.WriteLine("Total number of entries: {0}, Entries that are not 0: {1}", K.Length, nonZero);
#endif 
        }

        private void CalculateBoundaryMatrix()
        {
            // Clear boundary matrix
            B = new double[N_phi, N_lambda];

            // Define quadrature parameters
            double point1 = 0.5 - Math.Sqrt(3.0) / 6.0;
            double point2 = 0.5 + Math.Sqrt(3.0) / 6.0;

            double weight1 = 0.5;
            double weight2 = 0.5;

            double[] points = { point1, point2 };
            double[] weights = { weight1, weight2 };

            int quadraturePoints = points.Length;

            // Iterate through each boundary vertex
            foreach (Vertex boundaryVertex in Mesh.BoundaryVertices)
            {
                Vertex[] neighbourInternals = new Vertex[2];

                // Seraching for the two other internal vertices next to J-th boundary vertex
                for (int i = 0, l = 0; i < boundaryVertex.Neighbours.Count; i++)
                {
                    if (!boundaryVertex.Neighbours[i].IsBoundary)
                    {
                        neighbourInternals[l] = boundaryVertex.Neighbours[i];
                        l++;
                    }
                }

                // Count how many neighbours we found (1-2)
                int numbNeighbours = 0;
                for(int i = 0; i < neighbourInternals.Length; i++)
                {
                    if (neighbourInternals[i] != null)
                        numbNeighbours++;
                }

                // Calculate corresponding B_kj values
                for (int i = 0; i < numbNeighbours; i++)
                {
                    Vertex internalNeighbour = neighbourInternals[i];

                    // Creating edge, so it calculates sitance instantly
                    Edge edge = new Edge(new Vertex[]{ boundaryVertex, internalNeighbour }, false);

                    // Iterate through each boundary quadrature point along edge
                    for (int q = 0; q < quadraturePoints; q++)
                    {
                        // Parametric point
                        double xq = boundaryVertex.X + points[q] * (internalNeighbour.X - boundaryVertex.X);
                        double yq = boundaryVertex.Y + points[q] * (internalNeighbour.Y - boundaryVertex.Y);

                        // Search for element containing boundaryVertex & internalNeighbour vertices
                        Element? boundaryElement = Mesh.Elements.Find(x => ((x.V1 == boundaryVertex   ||
                                                                            x.V2 == boundaryVertex    ||
                                                                            x.V3 == boundaryVertex)    
                                                                            )&&(
                                                                            x.V1 == internalNeighbour || 
                                                                            x.V2 == internalNeighbour || 
                                                                            x.V3 == internalNeighbour));

                        if (boundaryElement != null)
                        {
                            // Global indexes of the boundary matrix
                            int j = boundaryVertex.BoundaryIndex;
                            int k = internalNeighbour.DomainIndex;

                            // Evaluate at which edge the boundary vertex and internal neighbour are connected with at the boundary element
                            int modeOfEvaluation = boundaryElement.GetEdgeIndex(boundaryElement.Edges.First(x => x.AreVerticesEqual(boundaryVertex, internalNeighbour)));

                            // Evaluate shapfe functions at the corresponding points

                            // phi_K
                            double phiK_XQ = EvaluateTentFunction(boundaryElement, modeOfEvaluation, xq, yq);
                            double psiJ_XQ = EvaluateBoundaryShapeFunction(boundaryElement, modeOfEvaluation, internalNeighbour.X - xq, internalNeighbour.Y - yq);

                            double lenght = edge.Length;

                            // Quadrature: w * phi_k(xq) * psi_j(xq) * |x_a - x_b|
                            B[k, j] += weights[q] * phiK_XQ * psiJ_XQ * lenght;
                        }
                        else
                        {
                            throw new ArgumentNullException(nameof(boundaryElement), "Element argument was null during shape function evaluation!");
                        }
                    }
                }
            }

#if _LOGGING_
            Console.WriteLine("The boundary matrix:");
            for (int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_lambda; j++)
                {
                    Console.Write("{0}\t", B[i, j].ToString("F1"));
                }
                Console.WriteLine();
            }

            int nonZero = 0;
            for(int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_lambda; j++)
                    if (B[i, j] != 0.0)
                        nonZero++;
            }

            Console.WriteLine("Total number of entries: {0}, Entries that are not 0: {1}", B.Length, nonZero);
#endif
        }

        private void CalculateLoadVector()
        {
            // Clear load vector
            f = new double[N_lambda];

            // Define quadrature parameters
            double point1 = 0.5 - Math.Sqrt(3.0) / 6.0;
            double point2 = 0.5 + Math.Sqrt(3.0) / 6.0;

            double weight1 = 0.5;
            double weight2 = 0.5;

            double[] points = { point1, point2 };
            double[] weights = { weight1, weight2 };

            int quadraturePoints = points.Length;

            foreach (Edge boundaryEdge in Mesh.BoundaryEdges)
            {
                Vertex V1 = boundaryEdge.Vertices[0];
                Vertex V2 = boundaryEdge.Vertices[1];

                // Find which element contains the edge
                Element? element = Mesh.Elements.Find(x => ((x.V1 == V1 || x.V2 == V1 || x.V3 == V1)
                                                         && (x.V1 == V2 || x.V2 == V2 || x.V3 == V2))
                                                         );

                if (element == null)
                    throw new ArgumentNullException(nameof(element), "Element argument was null during boundary data evaluation!");

                // Get which edge we are evaluation along
                int modeOfEvaluation = element.GetEdgeIndex(boundaryEdge);

                // Store the length of the edge for later calculations
                double length = boundaryEdge.Length;

                double boundaryData = 0.0;

                for (int q = 0; q < quadraturePoints; q++)
                {
                    // Parametric point
                    double xq = V1.X + points[q] * (V2.X - V1.X);
                    double yq = V1.Y + points[q] * (V2.Y - V1.Y);

                    // Evaluate boundary data at given point
                    if (element != null)
                    {
                        // Get the corresponding measurement value for the given edge

                        // Evaluate shape function in given point
                        double psi_JXQ = EvaluateBoundaryShapeFunction(element, modeOfEvaluation, xq, yq);

                        // Interpolate Phi at quadrature point
                        double v1Potential = BoundaryVoltages[V1.BoundaryIndex];
                        double v2Potential = BoundaryVoltages[V2.BoundaryIndex];

                        // Set current corresponding boundary data
                        boundaryData = v1Potential + points[q] * (v2Potential - v1Potential);

                        int j1 = V1.BoundaryIndex;
                        if (j1 >= 0)
                            f[j1] += weights[q] * boundaryData * psi_JXQ * length;
                        else 
                            throw new Exception("Boundary index is negative, on a boundary edge, check code!");

                        int j2 = V2.BoundaryIndex;
                        if (j2 >= 0)
                            f[j2] += weights[q] * boundaryData * psi_JXQ * length;
                        else
                            throw new Exception("Boundary index is negative, on a boundary edge, check code!");
                    }
                    else
                    {
                        throw new ArgumentNullException(nameof(element), "Element argument was null during shape function evaluation!");
                    }
                }
            }

#if _LOGGING_
            Console.WriteLine("The load vector f:");
            for (int j = 0; j < N_lambda; j++)
            {
                Console.WriteLine($"f[{j}] = {f[j]}");
            }
#endif
        }

        // Evaluate boundary shape function, currently same as EvaluateTentFunc
        private double EvaluateBoundaryShapeFunction(Element? element, int boundaryMode, double x, double y)
        {
            // Implement a separate evaluation for boundary shape functions if different from domain
            // For linear elements, boundary shape functions may coincide with domain shape functions restricted to the boundary
            // Hence, this could be the same as EvaluateTentFunction, but clarity is crucial
            return EvaluateTentFunction(element, boundaryMode, x, y);
        }

        // Evaluate tent function at specified point using barycentric coordinates
        // If a point lies
        public double EvaluateTentFunction(Element? element, int numMode, double x, double y)
        {
            if (element == null)
                throw new ArgumentNullException(nameof(element), "Can not evaluate shape function, null reference.");

            // Extract the triangle vertices
            double x1 = element.V1.X, y1 = element.V1.Y;
            double x2 = element.V2.X, y2 = element.V2.Y;
            double x3 = element.V3.X, y3 = element.V3.Y;

            double areaT = element.Area;

            // Compute sub-areas
            // A1 = area( (x,y), V2, V3 )
            double A1 = 0.5 * (x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2));

            // A2 = area( (x,y), V3, V1 )
            double A2 = 0.5 * (x * (y3 - y1) + x3 * (y1 - y) + x1 * (y - y3));

            // A3 = area( (x,y), V1, V2 )
            double A3 = 0.5 * (x * (y1 - y2) + x1 * (y2 - y) + x2 * (y - y1));

            // Barycentric coords: alpha = A1/areaT, beta = A2/areaT, gamma = A3/areaT
            double alpha = A1 / areaT;
            double beta = A2 / areaT;
            double gamma = A3 / areaT;

            // The shape function for node i is just alpha, beta, or gamma
            switch (numMode)
            {
                case 0: return alpha; // shape function for V1
                case 1: return beta;  // shape function for V2
                case 2: return gamma; // shape function for V3
                default: throw new ArgumentOutOfRangeException(nameof(numMode), "Value of i must be 0,1,2 for a linear triangle!");
            }
        }

        // Set the calulated potential values at the vertexes for visualization purposes
        private void SetPotentialValues(bool strongDirichlet)
        {
            // Clear previous potentials
            for (int i = 0; i < Mesh.InternalVertices.Count; i++)
                Mesh.InternalVertices[i].Potential = 0.0;

            // Set values
            for (int i = 0; i < Mesh.InternalVertices.Count; i++)
                Mesh.InternalVertices[i].Potential = Alpha[i];

            double avg = Alpha.Average();

            if(!strongDirichlet)
            {
                for (int i = 0; i < Mesh.BoundaryVertices.Count; i++)
                    Mesh.BoundaryVertices[i].Potential = Gamma[i];
            }
            else
            {
                for (int i = 0; i < Mesh.BoundaryVertices.Count; i++)
                    Mesh.BoundaryVertices[i].Potential = BoundaryVoltages[i];
            }
        }

        public void SetBoundaryPotentials(double[] boundaryPotentials)
        {
            if (boundaryPotentials.Count() != Mesh.BoundaryVertices.Count)
                throw new ArgumentOutOfRangeException(nameof(boundaryPotentials), "Size mismatch between boundary potentials and mesh boundary vertex count!");

            for (int i = 0; i < boundaryPotentials.Count(); i++)
                Mesh.BoundaryVertices[i].Potential = boundaryPotentials[i];
        }

        public void Solve(string? solverMethod)
        {
            if (String.IsNullOrEmpty(solverMethod))
                throw new ArgumentNullException(nameof(solverMethod), "Specified solver method is null, check code!");

#if _LOGGING_
            Console.WriteLine("\nCalculating Stiffness Matrix!\n");
#endif

            CalculateStiffnessMatrix();

#if _LOGGING_
            Console.WriteLine("\nCalculating Boundary Matrix!\n");
#endif

            CalculateBoundaryMatrix();

#if _LOGGING_
            Console.WriteLine("\nCalculating Load Vector!\n");
#endif
            CalculateLoadVector();

#if _LOGGING_
            Console.WriteLine("\nSolving the Saddle Point System!\n");
#endif
            if (solverMethod == "LU Decomposition")
                DiscreteSolver.Solve("LU", N_phi, N_lambda, K, B, f, ref Alpha, ref Gamma, -Theta, BoundaryVoltages);
            else if (solverMethod == "Singular Value Decomposition")
                DiscreteSolver.Solve("SVD", N_phi, N_lambda, K, B, f, ref Alpha, ref Gamma, -Theta, BoundaryVoltages);
            else if (solverMethod == "Conjugate Gradient")
                DiscreteSolver.Solve("CG", N_phi, N_lambda, K, B, f, ref Alpha, ref Gamma, -Theta, BoundaryVoltages);
            else if (solverMethod == "Without Lagrange Multipliers")
                DiscreteSolver.Solve("w.o.l.", N_phi, N_lambda, K, B, f, ref Alpha, ref Gamma, -Theta, BoundaryVoltages);
            else
                throw new ArgumentOutOfRangeException(nameof(solverMethod), "Solver method is not specified within possible values!");

            if (solverMethod == "Without Lagrange Multipliers")
                SetPotentialValues(true);
            else
                SetPotentialValues(false);
        }
    }
}
