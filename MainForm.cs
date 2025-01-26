using MathNet.Numerics.Statistics.Mcmc;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.Windows.Forms.VisualStyles;

namespace EIT_SOLVER
{
    public partial class MainForm : Form
    {
        private Mesh mesh = new Mesh();

        public MainForm()
        {
            InitializeComponent();
            InitializeCustomComponents();
        }

        // Initialize additional components and event handlers
        private void InitializeCustomComponents()
        {
            // Assign event handlers to buttons
            this.buttonGenerateRectangular.Click += ButtonGenerateRectangular_Click;
            this.buttonGenerateCircular.Click += ButtonGenerateCircular_Click;
            this.buttonSolve.Click += ButtonSolve_Click;

            // Initialize PictureBox settings
            this.pictureBoxMesh.BackColor = Color.White; // Background color for better visibility
            this.pictureBoxMesh.Paint += PictureBoxMesh_Paint;
        }

        // Event handler for generating rectangular mesh
        private void ButtonGenerateRectangular_Click(object sender, EventArgs e)
        {
            mesh = new Mesh();

            // Define mesh parameters
            double width = 20.0;
            double height = 10.0;

            ValidateAdnSetDensity(densityTextBox.Text.ToString());

            // Generate rectangular mesh
            mesh.GenerateMeshRectangular(width, height, mesh.Density);

            // Refresh the PictureBox to trigger repaint
            pictureBoxMesh.Invalidate();
        }

        // Event handler for generating circular mesh
        private void ButtonGenerateCircular_Click(object sender, EventArgs e)
        {
            mesh = new Mesh();

            // Define mesh parameters
            double radius = 10.0;
            ValidateAdnSetDensity(densityTextBox.Text.ToString());
            
            // Generate circular mesh
            mesh.GenerateMeshCircular(radius, mesh.Density);

            // Refresh the PictureBox to trigger repaint
            pictureBoxMesh.Invalidate();
        }

        // Event handler for solve button
        private void ButtonSolve_Click(object sender, EventArgs e)
        {
            double[] boundaryMeasurements = new double[mesh.BoundaryVertices.Count];

            // Circular inhomogenity in the middle enduces homogen boundary measurements in an ideal case
            for (int i = 0; i < boundaryMeasurements.Count(); i++)
                boundaryMeasurements[i] = 1.0;

            //boundaryMeasurements[mesh.BoundaryVertices.Count - 1] = 0.0;

            ForwardSolver forwardSolver = new ForwardSolver(mesh, boundaryMeasurements);

            forwardSolver.Solve();

            pictureBoxMesh.Invalidate();
        }

        // Checks if the string is a valid number, and within the specified range
        private void ValidateAdnSetDensity(string? densityString)
        {
            if(densityString != null && densityString != "")
            {
                if (densityString.Contains('.'))
                    densityString = densityString.Replace('.', ',');

                double density = Convert.ToDouble(densityString);

                if (density < 0)
                    mesh.Density = 0.3;
                if (density > 10)
                    mesh.Density = 1.0;
                else
                    mesh.Density = density;
            }
            else
                mesh.Density = 1.0;
        }

        // Paint event handler
        private void PictureBoxMesh_Paint(object sender, PaintEventArgs e)
        {
            if (mesh == null || mesh.Elements.Count == 0)
                return;

            Graphics g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            // 1) Determine bounding box and scale
            double minX = mesh.Vertices.Min(v => v.X);
            double maxX = mesh.Vertices.Max(v => v.X);
            double minY = mesh.Vertices.Min(v => v.Y);
            double maxY = mesh.Vertices.Max(v => v.Y);

            double meshWidth = maxX - minX;
            double meshHeight = maxY - minY;
            double padding = 20.0;

            double scaleX = (pictureBoxMesh.Width - 2 * padding) / meshWidth;
            double scaleY = (pictureBoxMesh.Height - 2 * padding) / meshHeight;
            double scale = Math.Min(scaleX, scaleY);

            double offsetX = padding + (pictureBoxMesh.Width - 2 * padding - meshWidth * scale) / 2.0;
            double offsetY = padding + (pictureBoxMesh.Height - 2 * padding - meshHeight * scale) / 2.0;

            PointF Transform(double x, double y)
            {
                float px = (float)((x - minX) * scale + offsetX);
                float py = (float)(pictureBoxMesh.Height - ((y - minY) * scale + offsetY));
                return new PointF(px, py);
            }

            // 2) Find min/max potential across vertices
            double minPot = mesh.Vertices.Min(v => v.Potential);
            double maxPot = mesh.Vertices.Max(v => v.Potential);

            // 3) For each element, fill with color
            foreach (var elem in mesh.Elements)
            {
                Vertex v1 = elem.V1;
                Vertex v2 = elem.V2;
                Vertex v3 = elem.V3;

                // average potential
                double avgPot = (v1.Potential + v2.Potential + v3.Potential) / 3.0;
                double frac = 0.0;
                if (maxPot > minPot)
                    frac = (avgPot - minPot) / (maxPot - minPot);

                // clamp [0..1]
                if (frac < 0) frac = 0;
                if (frac > 1) frac = 1;

                // e.g. blue(=0) -> red(=1)
                int R = (int)(255 * frac);
                int G = 0;
                int B = (int)(255 * (1.0 - frac));
                Color fillColor = Color.FromArgb(R, G, B);
                using (SolidBrush brush = new SolidBrush(fillColor))
                {
                    // triangle coords
                    PointF p1 = Transform(v1.X, v1.Y);
                    PointF p2 = Transform(v2.X, v2.Y);
                    PointF p3 = Transform(v3.X, v3.Y);
                    PointF[] pts = { p1, p2, p3 };

                    g.FillPolygon(brush, pts);
                }
            }

            // 4) Now draw edges on top
            using (Pen lightGrayPen = new Pen(Color.LightGray, 1))
            {
                foreach (var edge in mesh.InternalEdges)
                {
                    PointF p1 = Transform(edge.Vertices[0].X, edge.Vertices[0].Y);
                    PointF p2 = Transform(edge.Vertices[1].X, edge.Vertices[1].Y);
                    g.DrawLine(lightGrayPen, p1, p2);
                }
            }

            using (Pen blackPen = new Pen(Color.Black, 3))
            {
                foreach (var edge in mesh.BoundaryEdges)
                {
                    PointF p1 = Transform(edge.Vertices[0].X, edge.Vertices[0].Y);
                    PointF p2 = Transform(edge.Vertices[1].X, edge.Vertices[1].Y);
                    g.DrawLine(blackPen, p1, p2);
                }
            }
        }

    }
}
