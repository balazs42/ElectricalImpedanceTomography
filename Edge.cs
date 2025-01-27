using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EIT_SOLVER
{
    // An edge is represented between two vertices v1 ---- v2
    public class Edge
    {
        public Vertex[] Vertices = new Vertex[2];
        public double Length = 0.0;
        public bool IsBoundary = false;

        public Edge() { }
        public Edge(Vertex[] vertices, bool isBoundary)
        {
            Vertices = vertices;
            IsBoundary = isBoundary;

            Length = Math.Sqrt(Math.Pow((Vertices[0].X - Vertices[1].X), 2) +
                               Math.Pow((Vertices[0].Y - Vertices[1].Y), 2));
        }

        public Edge(Vertex v1, Vertex v2, bool isBoundary)
        {
            Vertices[0] = v1;
            Vertices[1] = v2;
            IsBoundary = isBoundary;

            Length = Math.Sqrt(Math.Pow((Vertices[0].X - Vertices[1].X), 2) +
                               Math.Pow((Vertices[0].Y - Vertices[1].Y), 2));
        }

        public bool AreVerticesEqual(Vertex v1, Vertex v2)
        {
            return (Vertices[0] == v1 && Vertices[1] == v2) || (Vertices[0] == v2 && Vertices[1] == v1);
        }
    }
}
