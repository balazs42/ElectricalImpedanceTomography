// ForwardSolver.cs 
#define _LOGGING_

using EIT_SOLVER;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Reflection.Metadata.Ecma335;
using System.Windows.Forms.VisualStyles;
using System.Xml.Linq;
using TriangleNet.Geometry;

namespace EIT_SOLVER
{
    public class ForwardSolver
    {
        public Mesh Mesh { get; }   // Mesh object holds all data for finite-element discretization
        public double[] BoundaryVoltages { get; }   // Measured voltages on the boundary
        private double[,] K { get; set; }   // Stiffness matrix
        private double[,] B { get; set; }   // Boundary matrix
        private double[] f { get; set; }    // Load vector
        public double[] Alpha { get; private set; } // Coefficients for φ
        private double[] Gamma { get; set; } // Coefficients for λ
        private int N_phi { get; set; } = 0;    // Number of φ coefficients
        private int N_lambda { get; set; } = 0; // Number of λ coefficients

        public ForwardSolver(Mesh mesh, double[] boundaryVoltages)
        {
            Mesh = mesh;
            BoundaryVoltages = boundaryVoltages;

            N_phi = Mesh.InternalVertices.Count();
            N_lambda = Mesh.BoundaryVertices.Count();

            // Creating matrices and vectors
            K = new double[N_phi, N_phi];
            B = new double[N_phi, N_lambda];
            f = new double[N_lambda];
            Alpha = new double[N_phi];
            Gamma = new double[N_lambda];
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

            //foreach(Vertex vertex in Mesh.Vertices)
            //{
            //    foreach(Vertex neighbourVertex in vertex.Neighbours)
            //    {
            //        Element? element = Mesh.Elements.Find(x => x.V1 == vertex || x.V2 == vertex || x.V3 == vertex);

            //        if (element != null)
            //        {
            //            if (vertex.DomainIndex == -1 || element.V1.DomainIndex == -1) continue;

            //            double sigma = element.Sigma;
            //            double area = element.Area;

            //            if (element.V1 == neighbourVertex)
            //            {
            //                double[,] gradPhi = element.GradPhi;

            //                double dotProd = (gradPhi[0, 0] * gradPhi[1, 0] + gradPhi[0, 1] * gradPhi[1, 1]);
            //                K[vertex.DomainIndex, element.V1.DomainIndex] += sigma * area * dotProd;

            //            }
            //            else if (element.V2 == neighbourVertex)
            //            {
            //                double[,] gradPhi = element.GradPhi;

            //                double dotProd = (gradPhi[1, 0] * gradPhi[1, 1] + gradPhi[2, 0] * gradPhi[2, 1]);
            //                K[vertex.DomainIndex, element.V1.DomainIndex] += sigma * area * dotProd;

            //            }
            //            else if (element.V3 == neighbourVertex)
            //            {
            //                double[,] gradPhi = element.GradPhi;

            //                double dotProd = (gradPhi[0, 0] * gradPhi[2, 0] + gradPhi[0, 1] * gradPhi[2, 1]);
            //                K[vertex.DomainIndex, element.V1.DomainIndex] += sigma * area * dotProd;
            //            }
            //        }

            //    }
            //}


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

                        // Evaluate shape function in given point
                        Element? boundaryElement = Mesh.Elements.Find(x => (x.V1 == boundaryVertex ||
                                                                            x.V2 == boundaryVertex ||
                                                                            x.V3 == boundaryVertex || 
                                                                            x.V1 == internalNeighbour || 
                                                                            x.V2 == internalNeighbour || 
                                                                            x.V3 == internalNeighbour));

                        if (boundaryElement != null)
                        {
                            // Global indexes of the boundary matrix
                            int j = boundaryVertex.BoundaryIndex;
                            int k = internalNeighbour.DomainIndex;

                            // Evaluate shape functions (in our case tent functions) using barycentric coordinates
                            double phiK_XQ = EvaluateTentFunction(boundaryElement, 0, xq, yq);
                            double psiJ_XQ = EvaluateTentFunction(boundaryElement, 0, xq, yq);

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

            //foreach (Edge boundaryEdge in Mesh.BoundaryEdges)
            //{
            //    Vertex V1 = boundaryEdge.Vertices[0];
            //    Vertex V2 = boundaryEdge.Vertices[1];

            //    // Iterate through each boundary quadrature point along edge
            //    for (int q = 0; q < quadraturePoints; q++)
            //    {
            //        // Parametric point
            //        double xq = V1.X + points[q] * (V2.X - V1.X);
            //        double yq = V1.Y + points[q] * (V2.Y - V1.Y);

            //        // Evaluate shape function in given point
            //        Element? boundaryElement = Mesh.Elements.Find(x => (x.Edges[0].IsBoundary || x.Edges[1].IsBoundary || x.Edges[2].IsBoundary));

            //        if (boundaryElement != null)
            //        {
            //            // Global indexes of the boundary matrix
            //            int k = V1.BoundaryIndex;
            //            int j = V2.BoundaryIndex;

            //            // Evaluate shape functions (in our case tent functions) using barycentric coordinates
            //            double phiK_XQ = EvaluateTentFunction(boundaryElement, 0, xq, yq);
            //            double psiJ_XQ = EvaluateTentFunction(boundaryElement, 0, xq, yq);

            //            double lenght = boundaryEdge.Length;

            //            // Quadrature: w * phi_k(xq) * psi_j(xq) * |x_a - x_b|
            //            B[k, j] += weights[q] * phiK_XQ * psiJ_XQ * lenght;
            //        }
            //        else
            //        {
            //            throw new ArgumentNullException(nameof(boundaryElement), "Element argument was null during shape function evaluation!");
            //        }
            //    }
            //}


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

            foreach(Edge boundaryEdge in Mesh.BoundaryEdges)
            {
                Vertex V1 = boundaryEdge.Vertices[0];
                Vertex V2 = boundaryEdge.Vertices[1];

                for(int q = 0; q < quadraturePoints; q++)
                {
                    // Parametric point
                    double xq = V1.X + points[q] * (V2.X - V1.X);
                    double yq = V1.Y + points[q] * (V2.Y - V1.Y);

                    // Evaluate boundary data at given point
                    double boundaryData = EvaluateBoundaryData(Mesh.Elements.Find(x => (x.Edges[0].IsBoundary || x.Edges[1].IsBoundary || x.Edges[2].IsBoundary)), 0, xq, yq);

                    // Evaluate the shape function for v1 on [v1,v2], which is (1 - t)
                    // Evaluate the shape function for v2 on [v1,v2], which is t
                    // Evaluate shape function in given point
                    Element? boundaryElement = Mesh.Elements.Find(x => (x.V1 == V1 ||
                                                                        x.V2 == V1 ||
                                                                        x.V3 == V1 ||
                                                                        x.V1 == V2 ||
                                                                        x.V2 == V2 ||
                                                                        x.V3 == V2));

                    if (boundaryElement != null)
                    {
                        double shape1 = EvaluateTentFunction(boundaryElement, 0, xq, yq);
                        double shape2 = EvaluateTentFunction(boundaryElement, 0, 1 - xq, 1 - yq);

                        int j1 = V1.BoundaryIndex;
                        if (j1 >= 0)
                        {
                            f[j1] += weights[q] * boundaryData * shape1 * boundaryEdge.Length;
                        }

                        int j2 = V2.BoundaryIndex;
                        if (j2 >= 0)
                        {
                            f[j2] += weights[q] * boundaryData * shape2 * boundaryEdge.Length;
                        }
                    }
                    else
                    {
                        throw new ArgumentNullException(nameof(boundaryElement), "Element argument was null during shape function evaluation!");
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

        // Evaluate boundary data at specified point, currently same functionality as in EvaluateTentFunction
        private double EvaluateBoundaryData(Element? element, int i, double x, double y)
        {
            return EvaluateTentFunction(element, i, x, y);
        }


        // Evaluate tent function at specified point using barycentric coordinates
        // If a point lies
        public double EvaluateTentFunction(Element? element, int i, double x, double y)
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
            double A1 = 0.5 * Math.Abs(
                x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)
            );
            // A2 = area( (x,y), V3, V1 )
            double A2 = 0.5 * Math.Abs(
                x * (y3 - y1) + x3 * (y1 - y) + x1 * (y - y3)
            );
            // A3 = area( (x,y), V1, V2 )
            double A3 = 0.5 * Math.Abs(
                x * (y1 - y2) + x1 * (y2 - y) + x2 * (y - y1)
            );

            // Barycentric coords: alpha = A1/areaT, beta = A2/areaT, gamma = A3/areaT
            double alpha = A1 / areaT;
            double beta = A2 / areaT;
            double gamma = A3 / areaT;

            // The shape function for node i is just alpha, beta, or gamma
            switch (i)
            {
                case 0: return alpha; // shape function for V1
                case 1: return beta;  // shape function for V2
                case 2: return gamma; // shape function for V3
                default: throw new ArgumentOutOfRangeException(nameof(i), "i must be 0,1,2 for a linear triangle");
            }
        }

        private void SolveSystem()
        {
            // Create a big Matrix<double> of size (N_phi + N_lambda) x (N_phi + N_lambda).

            int size = N_phi + N_lambda;
            var bigA = DenseMatrix.Create(size, size, 0.0);
            var rhsVec = DenseVector.Create(size, 0.0);

            // Fill top-left block with K
            for (int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_phi; j++)
                {
                    bigA[i, j] = K[i, j];
                }
            }

            // Fill top-right block with B
            for (int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_lambda; j++)
                {
                    bigA[i, N_phi + j] = B[i, j];
                }
            }

            // Fill bottom-left block with B^T
            for (int i = 0; i < N_phi; i++)
            {
                for (int j = 0; j < N_lambda; j++)
                {
                    // B^T has size N_lambda x N_phi, so the block is (row=N_phi..end, col=0..N_phi-1)
                    bigA[N_phi + j, i] = B[i, j];
                }
            }

            // The bottom-right block is all 0
            // => bigA[N_phi.., N_phi..] = 0.  It's already zero by default constructor above.

            // Fill rhs = [0, f]
            for (int j = 0; j < N_lambda; j++)
            {
                rhsVec[N_phi + j] = f[j];
            }

            bigA = bigA + 0.01 * DenseMatrix.CreateIdentity(size);

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

        // Solve the system assuming strong Dirichlet data
        private void SolveSystemWithoutLagrangeMultipliers()
        {
            f = new double[N_phi];
            for (int i = 0; i < N_phi; i++)
                if(i < N_lambda)
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
            {
                Alpha[i] = solution[i];
            }
#if _LOGGING_
            Console.WriteLine("Solution alpha[]:");
            for (int i = 0; i < N_phi; i++)
            {
                Console.WriteLine($"  alpha[{i}] = {Alpha[i]}");
            }
#endif
        }

        // Set the calulated potential values at the vertexes for visualization purposes
        private void SetPotentialValues()
        {
            // Clear previous potentials
            for (int i = 0; i < Mesh.InternalVertices.Count; i++)
                Mesh.InternalVertices[i].Potential = 0.0;

            // Set values
            for (int i = 0; i < Mesh.InternalVertices.Count; i++)
                Mesh.InternalVertices[i].Potential = Alpha[i];

            double avg = Alpha.Average();

            for (int i = 0; i < Mesh.BoundaryVertices.Count; i++)
                Mesh.BoundaryVertices[i].Potential = avg;

            for(int i = 0; i < Mesh.Vertices.Count; i++)
            {
                if (Mesh.Vertices[i].IsBoundary)
                {
                    Vertex? vertex = Mesh.BoundaryVertices.Find(x => x.X == Mesh.Vertices[i].X && x.Y == Mesh.Vertices[i].Y);

                    if (vertex != null)
                        Mesh.Vertices[i].Potential = vertex.Potential;
                }
                else
                {
                    Vertex? vertex = Mesh.InternalVertices.Find(x => x.X == Mesh.Vertices[i].X && x.Y == Mesh.Vertices[i].Y);

                    if (vertex != null)
                        Mesh.Vertices[i].Potential = vertex.Potential;
                }
            }
        }

        public void Solve()
        {
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
            //SolveSystemWithoutLagrangeMultipliers();
            SolveSystem();
            
            SetPotentialValues();
        }
    }
}
