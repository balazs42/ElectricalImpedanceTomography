using MathNet.Numerics.Statistics.Mcmc;
using Microsoft.VisualBasic.Devices;
using System;
using System.Collections.Generic;
using System.DirectoryServices.ActiveDirectory;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.Windows.Forms.VisualStyles;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace EIT_SOLVER
{
    public partial class MainForm : Form
    {
        private Mesh mesh = new Mesh();

        // Panning related fields
        private bool isPanning = false;
        private Point lastMousePosition;
        private float panOffsetX = 0;
        private float panOffsetY = 0;

        private bool meshCreated { get; set; } = false;

        // Tooltip for displaying information
        private System.Windows.Forms.ToolTip tooltip = new System.Windows.Forms.ToolTip();

        public MainForm()
        {
            InitializeComponent();
            InitializeCustomComponents();

            tooltip.SetToolTip(buttonGenerateRectangular, "Generate a rectangular mesh for EIT analysis.");
            tooltip.SetToolTip(buttonGenerateCircular, "Generate a circular mesh for EIT analysis.");
            tooltip.SetToolTip(densityTextBox, "Set the mesh density (e.g., 1.0).");
            tooltip.SetToolTip(regularizationTextBox, "Set the regularization parameter (e.g., 0.01).");
            tooltip.SetToolTip(buttonSolve, "Solve the forward problem with the current mesh and parameters.");
            tooltip.SetToolTip(solverComboBox, "Select which method you want to solve the system with.");
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
            this.pictureBoxMesh.Resize += PictureBoxMesh_Resize;

            // Assign mouse event handlers for panning and interactions
            this.pictureBoxMesh.MouseDown += PictureBoxMesh_MouseDown;
            this.pictureBoxMesh.MouseMove += PictureBoxMesh_MouseMove;
            this.pictureBoxMesh.MouseUp += PictureBoxMesh_MouseUp;
            this.pictureBoxMesh.MouseClick += PictureBoxMesh_MouseClick;

            this.solverComboBox.SelectedValueChanged += SolverComboBox_SelectedValueChanged;
        }

        // Event handler for combo box
        private void SolverComboBox_SelectedValueChanged(object? sender, EventArgs e)
        {
            if (sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while changing solver type!");

            // Hide regularization option when assuming strong dirichlet data
            if (solverComboBox.SelectedItem?.ToString() == "Without Lagrange Multipliers")
            {
                regularizationTextBox.Visible = false;
                regularizationTextBox.Enabled = false;
                regularizationLabel.ForeColor = Color.FromArgb(102, 102, 102);
            }
            else
            {
                regularizationTextBox.Visible = true;
                regularizationTextBox.Enabled = true;
                regularizationLabel.ForeColor = Color.Black;
            }
        }

        // Resize event handler
        private void PictureBoxMesh_Resize(object? sender, EventArgs e)
        {
            if (sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while resizing!");

            // Invalidate the PictureBox to trigger a repaint
            pictureBoxMesh.Invalidate();
        }

        // Event handler for generating rectangular mesh
        private void ButtonGenerateRectangular_Click(object? sender, EventArgs e)
        {
            if(sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to generate rectangle!");

            mesh = new Mesh();

            // Define mesh parameters
            double width = 20.0;
            double height = 10.0;

            ValidateAndSetDensity(densityTextBox.Text.ToString());

            // Reset panning offsets
            panOffsetX = 0;
            panOffsetY = 0;

            // Generate rectangular mesh
            mesh.GenerateMeshRectangular(width, height, mesh.Density);

            meshCreated = true;

            // Refresh the PictureBox to trigger repaint
            pictureBoxMesh.Invalidate();
        }

        // Event handler for generating circular mesh
        private void ButtonGenerateCircular_Click(object? sender, EventArgs e)
        {
            if(sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to generate circle!");

            mesh = new Mesh();

            // Define mesh parameters
            double radius = 10.0;
            ValidateAndSetDensity(densityTextBox.Text.ToString());

            // Reset panning offsets
            panOffsetX = 0;
            panOffsetY = 0;

            // Generate circular mesh
            mesh.GenerateMeshCircular(radius, mesh.Density);

            meshCreated = true;

            // Refresh the PictureBox to trigger repaint
            pictureBoxMesh.Invalidate();
        }

        // Event handler for solve button
        private void ButtonSolve_Click(object? sender, EventArgs e)
        {
            if(sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to solve!");

            if (!meshCreated)
            {
                MessageBox.Show("Please generate a mesh first!", "Mesh not created", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            double[] boundaryMeasurements = new double[mesh.BoundaryVertices.Count];

            // Load measurement data
            Measurement measurement = new Measurement(mesh.BoundaryVertices.Count);

            if(simulationComboBox.SelectedItem?.ToString() != null)
            {
                string simulationType = simulationComboBox.SelectedItem.ToString();

                if (simulationType == "Eqvipotential")
                    boundaryMeasurements = measurement.SimulateEquvipotentialSurface(mesh.InternalVertices.Count, 2.0);
                else if (simulationType == "Assymtric")
                    boundaryMeasurements = measurement.SimulateAssymetricMeasurements(mesh.InternalVertices.Count, 2.0);
                else if (simulationType == "Noise")
                    boundaryMeasurements = measurement.SimulateNoise(mesh.InternalVertices.Count, 1.5, 2.0);
                else
                    throw new ArgumentOutOfRangeException(nameof(simulationType), "Simulation type was not recognized!");
            }

            // Denoise data 
            if(denoisingComboBox.SelectedItem.ToString() != null)
            {
                string denoiseType = denoisingComboBox.SelectedItem?.ToString();
                if (denoiseType == "None") { /*Do nothing*/}
                else if (denoiseType == "Moving average")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.MovingAverage, 3);
                else if (denoiseType == "Median")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.Median, 3);
                else if (denoiseType == "Gaussian")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.Gaussian, 3);
                else if (denoiseType == "Kalman")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.Kalman, 3);
                else if (denoiseType == "Wavelet")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.Wavelet, 3);
                else if (denoiseType == "Wiener")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.Wiener, 3);
                else if (denoiseType == "Savitzky-Golay")
                    boundaryMeasurements = measurement.DenoiseMeasurement(boundaryMeasurements, Measurement.DenoisingMethod.SavitzkyGolay, 3);
                else
                    throw new ArgumentOutOfRangeException(nameof(denoiseType), "Denoising type was not recognized!");
            }

            ForwardSolver forwardSolver = new ForwardSolver(mesh, boundaryMeasurements);

            double theta = ValidateAndSetRegularizationParameter(regularizationTextBox.Text.ToString());

            forwardSolver.Theta = theta;

            string? solverType = solverComboBox.SelectedItem?.ToString();
            if (solverType == null)
                throw new ArgumentNullException(nameof(solverType), "Solver type was null while trying to extract from combobox!");

            forwardSolver.Solve(solverType);

            pictureBoxMesh.Invalidate();
        }

        // MouseDown event to start panning
        private void PictureBoxMesh_MouseDown(object? sender, MouseEventArgs e)
        {
            if(!meshCreated)
                return;

            if(sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to pan!");

            if (e.Button == MouseButtons.Left)
            {
                isPanning = true;
                lastMousePosition = e.Location;
                this.pictureBoxMesh.Cursor = Cursors.Hand;
            }
        }

        // MouseMove event to handle panning and hovering
        private void PictureBoxMesh_MouseMove(object? sender, MouseEventArgs e)
        {
            if(sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to pan or hover!");

            if (isPanning)
            {
                // Calculate the movement
                float dx = e.X - lastMousePosition.X;
                float dy = e.Y - lastMousePosition.Y;

                // Update pan offsets
                panOffsetX += dx;
                panOffsetY += dy;

                // Update last mouse position
                lastMousePosition = e.Location;

                // Redraw the mesh
                pictureBoxMesh.Invalidate();
            }
            else
            {
                // Handle hovering to show tooltips
                HandleHover(e.Location);
            }
        }

        private void HandleHover(Point mouseLocation)
        {
            if (!meshCreated)
                return;

            // Map mouse location to mesh coordinates
            var meshPoint = ScreenToMesh(mouseLocation);

            // Define a tolerance in mesh units (e.g., 0.2 units)
            double tolerance = mesh.Density * 0.2;

            // Check for vertex hover
            foreach (Vertex vertex in mesh.Vertices)
            {
                double distance = Math.Sqrt(Math.Pow(vertex.X - meshPoint.X, 2) + Math.Pow(vertex.Y - meshPoint.Y, 2));
                if (distance <= tolerance)
                {
                    if (vertex.IsBoundary)
                    {
                        // Show tooltip for boundary vertex
                        string vertexInfo = $"Vertex Boundary ID: {vertex.BoundaryIndex}\nCoordinates: ({vertex.X:F2}, {vertex.Y:F2})\nPotential: {vertex.Potential:F2}";
                        tooltip.Show(vertexInfo, pictureBoxMesh, mouseLocation.X + 10, mouseLocation.Y + 10, 1000);
                        return;
                    }
                    else
                    {
                        // Show tooltip for internal vertex
                        string vertexInfo = $"Vertex Internal ID: {vertex.DomainIndex}\nCoordinates: ({vertex.X:F2}, {vertex.Y:F2})\nPotential: {vertex.Potential:F2}";
                        tooltip.Show(vertexInfo, pictureBoxMesh, mouseLocation.X + 10, mouseLocation.Y + 10, 1000);
                        return;
                    }
                }
            }

            // Check for element hover
            foreach (var elem in mesh.Elements)
            {
                if (IsPointInTriangle(meshPoint, elem.V1, elem.V2, elem.V3))
                {
                    // Show tooltip for element
                    string elemInfo = $"Element Area: {elem.Area:F2}\nSigma: {elem.Sigma:F2}";
                    tooltip.Show(elemInfo, pictureBoxMesh, mouseLocation.X + 10, mouseLocation.Y + 10, 1000);
                    return;
                }
            }

            // If not hovering over anything, hide tooltip
            tooltip.Hide(pictureBoxMesh);
        }

        // Convert screen (pixel) coordinates to mesh coordinates
        private (double X, double Y) ScreenToMesh(PointF screenPoint)
        {
            // Determine bounding box and scale as in Paint event
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

            double transformedX = (screenPoint.X - padding - (pictureBoxMesh.Width - 2 * padding - meshWidth * scale) / 2.0 - panOffsetX) / scale + minX;
            double transformedY = ((pictureBoxMesh.Height - screenPoint.Y - padding - (pictureBoxMesh.Height - 2 * padding - meshHeight * scale) / 2.0) - panOffsetY) / scale + minY;

            return (transformedX, transformedY);
        }

        // Check if a point is inside a triangle using barycentric coordinates
        private bool IsPointInTriangle((double X, double Y) pt, Vertex v1, Vertex v2, Vertex v3)
        {
            double denominator = ((v2.Y - v3.Y) * (v1.X - v3.X) + (v3.X - v2.X) * (v1.Y - v3.Y));
            double a = ((v2.Y - v3.Y) * (pt.X - v3.X) + (v3.X - v2.X) * (pt.Y - v3.Y)) / denominator;
            double b = ((v3.Y - v1.Y) * (pt.X - v3.X) + (v1.X - v3.X) * (pt.Y - v3.Y)) / denominator;
            double c = 1 - a - b;

            // Allow points on the edges by using >= and <=
            return a >= 0 && b >= 0 && c >= 0;
        }

        // MouseUp event to stop panning
        private void PictureBoxMesh_MouseUp(object? sender, MouseEventArgs e)
        {
            if (sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to stop panning!");

            if (e.Button == MouseButtons.Left)
            {
                isPanning = false;
                this.pictureBoxMesh.Cursor = Cursors.Default;
            }
        }

        // MouseClick event handler
        private void PictureBoxMesh_MouseClick(object? sender, MouseEventArgs e)
        {
            if(!meshCreated)
                return;

            if (sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to click on mesh!");

            if (e.Button == MouseButtons.Left)
            {
                // Map mouse location to mesh coordinates
                var meshPoint = ScreenToMesh(e.Location);

                int index = 0;
                // Find the element under the mouse
                foreach (var elem in mesh.Elements)
                {
                    if (IsPointInTriangle(meshPoint, elem.V1, elem.V2, elem.V3))
                    {
                        // Modify the sigma value (e.g., increase by 10%)
                        elem.Sigma *= 1.1;

                        // Optionally, cap the sigma value to prevent it from becoming too large
                        elem.Sigma = Math.Min(elem.Sigma, 1000.0); // Example cap

                        // Redraw the mesh to reflect changes
                        pictureBoxMesh.Invalidate();

                        // Update the tooltip to show the new sigma value
                        HandleHover(e.Location);

                        // Optionally, show a message or update the UI
                        MessageBox.Show($"Element Index: {index} sigma increased to {elem.Sigma:F2}", "Sigma Updated", MessageBoxButtons.OK, MessageBoxIcon.Information);

                        // Exit after handling the first matching element
                        return;
                    }
                    index++;
                }
            }
        }

        // Checks if the string is a valid number, and within the specified range
        private void ValidateAndSetDensity(string? densityString)
        {
            if (densityString != null && densityString != "")
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

        // Checks if the string is a valid number, and within the specified range
        private double ValidateAndSetRegularizationParameter(string? theta)
        {
            if (theta != null && theta != "")
            {
                if (theta.Contains('.'))
                    theta = theta.Replace('.', ',');

                double tht = Convert.ToDouble(theta);

                if (tht < 0.0)
                    return 1.0;
                if (tht > 10.0)
                    return 10.0;
                else
                    return tht;
            }
            else
                return 1.0;
        }

        // Paint event handler
        private void PictureBoxMesh_Paint(object? sender, PaintEventArgs e)
        {
            if (mesh == null || mesh.Elements.Count == 0)
                return;

            if(sender == null)
                throw new ArgumentNullException(nameof(sender), "Sender was null while trying to paint mesh!");

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

            // Inside PictureBoxMesh_Paint method
            PointF Transform(double x, double y)
            {
                float px = (float)((x - minX) * scale + offsetX + panOffsetX);
                float py = (float)(pictureBoxMesh.Height - ((y - minY) * scale + offsetY) + panOffsetY);
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
