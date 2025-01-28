// Vertex.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EIT_SOLVER
{
    public class Vertex
    {
        public double X { get; set; }
        public double Y { get; set; }
        public bool IsBoundary { get; set; }
        public bool IsMeasurement { get; set; } = false;
        public int DomainIndex { get; set; } = -1;
        public int BoundaryIndex { get; set; } = -1;
        public double Potential { get; set; } = 0;
        public List<Vertex> Neighbours { get; set; }
        public List<Edge> Edges { get; set; }

        public Vertex(double x, double y, bool isBoundary)
        {
            X = x;
            Y = y;
            IsBoundary = isBoundary;
            Neighbours = new List<Vertex>();
            Edges = new List<Edge>();
        }

        public Vertex(double x, double y, bool isBoundary, bool isMeasurement)
        {
            X = x;
            Y = y;
            IsBoundary = isBoundary;
            Neighbours = new List<Vertex>();
            Edges = new List<Edge>();

            if (isBoundary && isMeasurement)
                isMeasurement = true;
            else
                isMeasurement = false;
        }


        public double Distance()
        {
            return Math.Sqrt(X * X + Y * Y);
        }

        public double Distance(Vertex v)
        {
            return Math.Sqrt(Math.Pow(X - v.X, 2) + Math.Pow(Y - v.Y, 2));
        }

        public double Distance(double x, double y)
        {
            return Math.Sqrt(Math.Pow(X - x, 2) + Math.Pow(Y - y, 2));
        }

        public int GetEdgeIndex(Edge whatEdge)
        {
            Vertex v1 = whatEdge.Vertices[0];
            Vertex v2 = whatEdge.Vertices[1];

            for (int i = 0; i < Edges.Count(); i++)
                if ((Edges[i].Vertices[0] == v1 && Edges[i].Vertices[0] == v2) ||
                    (Edges[i].Vertices[0] == v2 && Edges[i].Vertices[1] == v1))
                    return i;

            return -1;
        }
    }
}

