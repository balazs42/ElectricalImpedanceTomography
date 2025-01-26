// Vertex.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MIConvexHull;

namespace EIT_SOLVER
{
    public class Vertex
    {
        public double X { get; set; }
        public double Y { get; set; }
        public bool IsBoundary { get; }
        public bool IsMeasurement { get; set; } = false;
        public int DomainIndex { get; set; } = -1;
        public int BoundaryIndex { get; set; } = -1;
        public double Potential { get; set; } = -10000;
        public Vertex[] Neighbours { get; set; }

        public Vertex(double x, double y, bool isBoundary)
        {
            X = x;
            Y = y;
            IsBoundary = isBoundary;
            Neighbours = new Vertex[30];
        }

        public Vertex(double x, double y, bool isBoundary, bool isMeasurement)
        {
            X = x;
            Y = y;
            IsBoundary = isBoundary;
            if (isBoundary && isMeasurement)
                isMeasurement = true;
            else
                isMeasurement = false;
        }


        public double Distance()
        {
            return Math.Sqrt(X * X + Y * Y);
        }
    }
}

