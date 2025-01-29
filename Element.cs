using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Element.cs

namespace EIT_SOLVER
{
    public class Element
    {
        public Vertex V1 { get; }
        public Vertex V2 { get; }
        public Vertex V3 { get; }
        public Edge[] Edges { get; } = new Edge[3];
        public double Area { get; private set; }
        public double[,] GradPhi { get; private set; } // Gradients of shape functions
        public double Sigma { get; set; } = 1.0; // Conductivity
        public double[,] DotProducts { get; private set; } = new double[3, 3];
        public int Id { get; set; } = -1;

        public Element(Vertex v1, Vertex v2, Vertex v3, int id)
        {
            V1 = v1;
            V2 = v2;
            V3 = v3;

            Id = id;

            Edges[0] = (v1.IsBoundary && v2.IsBoundary) ? new Edge(v1, v2, true) : new Edge(v1, v2, false);
            Edges[1] = (v1.IsBoundary && v3.IsBoundary) ? new Edge(v1, v3, true) : new Edge(v1, v3, false);
            Edges[2] = (v2.IsBoundary && v3.IsBoundary) ? new Edge(v2, v3, true) : new Edge(v2, v3, false);

            GradPhi = new double[3, 2];

            // Calculate area of given element
            CalculateArea();

            // Calculate shape function gradients on the given element
            CalculateGradients();

            // Calculate the dot product of the gradients of shape functions on given element
            CalculateDotProducts();

            if (GradPhi == null || DotProducts == null)
                throw new InvalidDataException("GradPhi or DotProducts was null during element initialization, check code!");
        }

        private void InitializeEdges()
        {
            Edges[0] = new Edge(V1, V2, false);
            Edges[1] = new Edge(V2, V3, false);
            Edges[2] = new Edge(V3, V1, false);
        }

        private void CalculateArea()
        {
            // Using the shoelace formula
            Area = 0.5 * Math.Abs(
                V1.X * (V2.Y - V3.Y) +
                V2.X * (V3.Y - V1.Y) +
                V3.X * (V1.Y - V2.Y));
        }

        private void CalculateGradients()
        {
            // Gradients of the linear shape functions are constant within the element
            double x1 = V1.X, y1 = V1.Y;
            double x2 = V2.X, y2 = V2.Y;
            double x3 = V3.X, y3 = V3.Y;

            GradPhi = new double[3, 2];

            // ∇ϕ₁
            GradPhi[0, 0] = (y2 - y3) / (2.0 * Area);  // d/dx
            GradPhi[0, 1] = (x3 - x2) / (2.0 * Area);  // d/dy

            // ∇ϕ₂
            GradPhi[1, 0] = (y3 - y1) / (2.0 * Area);
            GradPhi[1, 1] = (x1 - x3) / (2.0 * Area);

            // ∇ϕ₃
            GradPhi[2, 0] = (y1 - y2) / (2.0 * Area);
            GradPhi[2, 1] = (x2 - x1) / (2.0 * Area);
        }

        private void CalculateDotProducts()
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    double dotPorduct = GradPhi[i, 0] * GradPhi[j, 0] +
                                        GradPhi[i, 1] * GradPhi[j, 1];

                    DotProducts[i, j] = dotPorduct;
                }
            }
        }

        public int GetEdgeIndex(Edge whatEdge)
        {
            for(int i = 0; i < Edges.Length; i++)
                if (Edges[i].Equals(whatEdge))
                    return i;
  
            return -1;
        }
    }
}

