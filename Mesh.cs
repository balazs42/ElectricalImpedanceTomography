// Mesh.cs
using System;
using System.Collections.Generic;
using MIConvexHull;
using System.Linq;
using TriangleNet.Geometry;
using TriangleNet.Meshing;
using System.Windows.Forms.VisualStyles;
using MathNet.Numerics.Optimization;

namespace EIT_SOLVER
{
    public class Mesh
    {
        public List<Vertex> Vertices { get; }
        public List<Vertex> InternalVertices { get; private set; }
        public List<Vertex> BoundaryVertices { get; private set; }
        public List<Element> Elements { get; }
        public List<Edge> BoundaryEdges { get; private set; }
        public List<Edge> InternalEdges { get; private set; }
        public double Density { get; set; } = 1.0;
        public double Size { get; set; } = 10;
        public Mesh()
        {
            Vertices = new List<Vertex>();
            InternalVertices = new List<Vertex>();
            BoundaryVertices = new List<Vertex>();
            Elements = new List<Element>();
            BoundaryEdges = new List<Edge>();
            InternalEdges = new List<Edge>();
        }

        public Mesh(List<Vertex> vertices, List<Element> elements)
        {
            Vertices = vertices;
            Elements = elements;
        }

        // Generate mesh using Delaunay triangulation
        // vertices describes the boundary of the mesh and the points inside the boundary
        // Delaunay triangulation using Triangle.NET
        public void GenerateMeshDelaunay(List<Vertex> inputVertices, double minAngle = 20.0)
        {
            // Clear existing data
            Vertices.Clear();
            Elements.Clear();
            BoundaryEdges.Clear();
            InternalEdges.Clear();

            // Assign unique IDs to vertices if needed
            foreach (var v in inputVertices)
            {
                v.X += Size;
                v.Y += Size;
                Vertices.Add(v);
            }

            // Create a Triangle.NET Polygon
            var polygon = new Polygon();

            // Assuming the first four vertices form the boundary (for rectangular)
            // Modify as needed based on your mesh generation
            foreach (var v in inputVertices)
            {
                polygon.Add(new TriangleNet.Geometry.Vertex(v.X, v.Y));
            }

            // Define meshing options
            var options = new ConstraintOptions
            {
                ConformingDelaunay = true,
                Convex = true
            };
            var quality = new QualityOptions
            {
                MinimumAngle = minAngle // Adjust as needed
            };

            // Generate the mesh
            var meshNet = polygon.Triangulate(options, quality);

            // Add elements to the mesh
            foreach (var tri in meshNet.Triangles)
            {
                var v1 = Vertices.FirstOrDefault(v => v.X == tri.GetVertex(0).X && v.Y == tri.GetVertex(0).Y);
                var v2 = Vertices.FirstOrDefault(v => v.X == tri.GetVertex(1).X && v.Y == tri.GetVertex(1).Y);
                var v3 = Vertices.FirstOrDefault(v => v.X == tri.GetVertex(2).X && v.Y == tri.GetVertex(2).Y);
                
                Vertex V1 = new Vertex(v1.X + Size, v1.Y + Size, false);
                Vertex V2 = new Vertex(v2.X + Size, v2.Y + Size, false); 
                Vertex V3 = new Vertex(v3.X + Size, v3.Y + Size, false); 

                if (v1 != null && v2 != null && v3 != null)
                {
                    Element element = new Element(v1, v2, v3);
                    Elements.Add(element);
                }
            }

            // Identify boundary edges
            IdentifyBoundaryEdges();

            foreach(Vertex vertex in Vertices)
            {
                if (vertex.IsBoundary)
                    BoundaryVertices.Add(vertex);
                else
                    InternalVertices.Add(vertex);
            }

            // Assign global indexes to vertices
            AssignGlobalIndexes();

            // Log the created mesh
            LogMesh();

            // Assign neighbours
            AssignNeighbours();
        }

        // Generate 2d meshes
        public void GenerateMeshRectangular(double width, double height, double density = 0.1)
        {
            if(width > 20.0) width = 20.0;   // Limit the width for visualization
            if(height > 20.0) height = 20.0; // Limit the height for visualization

            List<Vertex> vertices = new List<Vertex>();

            // Generate boundary vertices (rectangle corners)
            vertices.Add(new Vertex(0.0, 0.0, true));
            vertices.Add(new Vertex(width, 0.0, true));
            vertices.Add(new Vertex(width, height, true));
            vertices.Add(new Vertex(0.0, height, true));

            // Calculate the number of points along each axis based on density
            int numPointsX = (int)(width / density) + 1;
            int numPointsY = (int)(height / density) + 1;

            // Generate interior vertices
            for (int i = 1; i < numPointsX - 1; i++)
            {
                for (int j = 1; j < numPointsY - 1; j++)
                {
                    double x = i * density;
                    double y = j * density;
                    vertices.Add(new Vertex(x, y, false));
                }
            }

            // Optionally, add boundary points along the edges for better mesh quality
            // Left and right edges
            for (int j = 1; j < numPointsY; j++)
            {
                double y = j * density;
                vertices.Add(new Vertex(0.0, y, true));
                vertices.Add(new Vertex(width, y, true));
            }

            // Top and bottom edges
            for (int i = 1; i < numPointsX; i++)
            {
                double x = i * density;
                vertices.Add(new Vertex(x, 0.0, true));
                vertices.Add(new Vertex(x, height, true));
            }

            // Remove duplicate vertices
            vertices = RemoveDuplicateVertices(vertices);

            // Call Delaunay triangulation
            GenerateMeshDelaunay(vertices, 40.0);
        }
        public void GenerateMeshCircular(double radius, double density = 1.0)
        {
            List<Vertex> vertices = new List<Vertex>();

            // Add the central vertex
            vertices.Add(new Vertex(0.0, 0.0, false));

            // Number of points on the boundary based on circumference and density
            int numBoundaryPoints = (int)(2 * Math.PI * radius / density);
            numBoundaryPoints = Math.Max(numBoundaryPoints, 20); // Minimum number for smoothness

            // Generate boundary vertices (approximated circle)
            for (int i = 0; i < numBoundaryPoints; i++)
            {
                double angle = 2 * Math.PI * i / numBoundaryPoints;
                double x = radius * Math.Cos(angle);
                double y = radius * Math.Sin(angle);
                vertices.Add(new Vertex(x, y, true));
            }

            // Calculate the number of points in radial and angular directions based on density
            int numRadial = (int)(radius / density);
            int numAngular = numBoundaryPoints;

            // Generate interior vertices
            for (int r = 0; r < numRadial; r++)
            {
                double currentRadius = (r + 1) * density;
                numAngular = (int)(2 * Math.PI * currentRadius / density);
                for (int i = 0; i < numAngular; i++)
                {
                    double angle = 2 * Math.PI * i / numAngular;
                    double x = (currentRadius) * Math.Cos(angle);
                    double y = (currentRadius) * Math.Sin(angle);
                    vertices.Add(new Vertex(x, y, false));
                }
            }

            // Remove duplicate vertices
            vertices = RemoveDuplicateVertices(vertices, tolerance: 1e-9);

            // Call Delaunay triangulation
            GenerateMeshDelaunay(vertices);
        }


        private void IdentifyBoundaryEdges()
        {
            // Search for greates distance from middle point
            double maxDist = 0.0;
            foreach(var element in Elements)
            {
                double dist = Math.Max(element.V1.Distance(), element.V2.Distance());
                dist = Math.Max(dist, element.V3.Distance());
                if (dist > maxDist)
                    maxDist = dist;
            }


            foreach (var element in Elements)
            {
                foreach(var edge in element.Edges)
                {
                    if (edge.IsBoundary)
                    {
                        var v1 = Vertices.FirstOrDefault(v => v.X == edge.Vertices[0].X && v.Y == edge.Vertices[0].Y);
                        var v2 = Vertices.FirstOrDefault(v => v.X == edge.Vertices[1].X && v.Y == edge.Vertices[1].Y);

                        if (v1 != null && v2 != null)
                        {
                            BoundaryEdges.Add(new Edge(v1, v2, true));
                            
                            // Setting corresponging element's edge as boundary edge
                            Element? boundaryElement = Elements.Find(x => x.Edges[0] == edge || x.Edges[1] == edge || x.Edges[2] == edge);
                            if (boundaryElement != null)
                            {
                                Edge boundaryEdge = boundaryElement.Edges.First(e => e.IsBoundary);
                                boundaryEdge.IsBoundary = true;
                            }
                        }
                    }
                    else
                    {
                        var v1 = Vertices.FirstOrDefault(v => v.X == edge.Vertices[0].X && v.Y == edge.Vertices[0].Y);
                        var v2 = Vertices.FirstOrDefault(v => v.X == edge.Vertices[1].X && v.Y == edge.Vertices[1].Y);

                        if (v1 != null && v2 != null)
                        {
                            InternalEdges.Add(new Edge(v1, v2, false));
                        }
                    }
                }
            }
        }

        private List<Vertex> RemoveDuplicateVertices(List<Vertex> vertices, double tolerance = 1e-6)
        {
            return vertices
                .GroupBy(v => new { X = Math.Round(v.X / tolerance) * tolerance, Y = Math.Round(v.Y / tolerance) * tolerance })
                .Select(g => g.First())
                .ToList();
        }

        private void AssignGlobalIndexes()
        {
            // Assign domain (bulk) shape function indices
            int N_phi = InternalVertices.Count;

            for (int i = 0; i < N_phi; i++)       
                InternalVertices[i].DomainIndex = i;
            
            // 3) Assign boundary shape function indices
            int N_lambda = BoundaryVertices.Count;

            for (int j = 0; j < N_lambda; j++)            
                BoundaryVertices[j].BoundaryIndex = j;
            
        }

        private void LogMesh()
        {
            int i = 0;

            Console.WriteLine("Vertexes:");
            // Iterate through each vertex and log their data
            foreach(Vertex vertex in Vertices)
            {
                Console.WriteLine("Vertex {0}: X = {1}, Y = {2}, Domain Index = {3}, Boundary Index = {4}", i, vertex.X, vertex.Y, vertex.DomainIndex, vertex.BoundaryIndex);
                i++;
            }

            i = 0;

            Console.WriteLine("Boundary Edges:");
            foreach(Edge edge in BoundaryEdges)
            {
                Console.WriteLine("Edge {0}: From:({1},{2}) to ({3},{4}), Length = {5}, IsBoundary = {6}", i, edge.Vertices[0].X, edge.Vertices[0].Y, edge.Vertices[1].X, edge.Vertices[1].Y, edge.Length, edge.IsBoundary);
                i++;
            }

            i = 0;

            Console.WriteLine("Internal Edges:");
            foreach (Edge edge in InternalEdges)
            {
                Console.WriteLine("Edge {0}: From:({1},{2}) to ({3},{4}), Length = {5}, IsBoundary = {6}", i, edge.Vertices[0].X, edge.Vertices[0].Y, edge.Vertices[1].X, edge.Vertices[1].Y, edge.Length, edge.IsBoundary);
                i++;
            }
        }

        private void AssignNeighbours()
        {
            int i = 0;
            foreach(Vertex vertex in Vertices)
            {
                i = 0;
                foreach(Edge edge in InternalEdges)
                {
                    if (edge.Vertices[0].X == vertex.X && edge.Vertices[0].Y == vertex.Y)
                    {
                        vertex.Neighbours[i] = edge.Vertices[1];
                        i++;
                    }
                    else if (edge.Vertices[1].X == vertex.X && edge.Vertices[1].Y == vertex.Y)
                    {
                        vertex.Neighbours[i] = edge.Vertices[0];
                        i++;
                    }
                }
                i = 0;
                foreach (Edge edge in BoundaryEdges)
                {
                    if (edge.Vertices[0].X == vertex.X && edge.Vertices[0].Y == vertex.Y)
                    {
                        vertex.Neighbours[i] = edge.Vertices[1];
                        i++;
                    }
                    else if (edge.Vertices[1].X == vertex.X && edge.Vertices[1].Y == vertex.Y)
                    {
                        vertex.Neighbours[i] = edge.Vertices[0];
                        i++;
                    }
                }

            }
        }
    }
}
