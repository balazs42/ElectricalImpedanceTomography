// MainForm.cs
using System;
using System.Drawing;
using System.Windows.Forms;

namespace EIT_SOLVER
{
    partial class MainForm
    {
        private System.ComponentModel.IContainer components = null;

        // --- PictureBoxes ---
        private PictureBox pictureBoxPotential; // was pictureBoxMesh
        private PictureBox pictureBoxSigma;     // new: for visualizing sigma

        // --- Buttons ---
        private Button buttonGenerateRectangular;
        private Button buttonGenerateCircular;
        private Button buttonForwardSolve;  // was buttonSolve
        private Button buttonInverseSolve;  // new

        // --- Other Controls ---
        private TextBox densityTextBox;
        private Label densityLabel;
        private TextBox regularizationTextBox;
        private Label regularizationLabel;
        private Label solverLabel;
        private ComboBox solverComboBox;
        private Label simulationLabel;
        private ComboBox simulationComboBox;
        private Label denoisingLabel;
        private ComboBox denoisingComboBox;

        private Label labelPotential; // Label for Potential Distribution
        private Label labelSigma;     // Label for Conductivity Distribution

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();

            // 1) Instantiate PictureBoxes
            pictureBoxPotential = new PictureBox();
            pictureBoxSigma = new PictureBox();

            // 2) Instantiate Buttons
            buttonGenerateRectangular = new Button();
            buttonGenerateCircular = new Button();
            buttonForwardSolve = new Button();
            buttonInverseSolve = new Button();

            // 3) Instantiate other controls
            densityLabel = new Label();
            densityTextBox = new TextBox();
            regularizationLabel = new Label();
            regularizationTextBox = new TextBox();
            solverLabel = new Label();
            solverComboBox = new ComboBox();
            simulationLabel = new Label();
            simulationComboBox = new ComboBox();
            denoisingLabel = new Label();
            denoisingComboBox = new ComboBox();

            // solverComboBox items
            solverComboBox.Items.AddRange(new object[] {
            "LU Decomposition",
            "Singular Value Decomposition",
            "Truncated SVD",
            "Conjugate Gradient",
            "Without Lagrange Multipliers",
            "GMRES" });
            solverComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            solverComboBox.SelectedIndex = 0;

            // simulationComboBox items
            simulationComboBox.Items.AddRange(new object[] {
            "Eqvipotential",
            "Gaussian",
            "Assymtric",
            "Noise" });
            simulationComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            simulationComboBox.SelectedIndex = 0;

            // denoisingComboBox items
            denoisingComboBox.Items.AddRange(new object[] {
            "None",
            "Moving average",
            "Median",
            "Gaussian",
            "Kalman",
            "Wavelet",
            "Wiener",
            "Savitzky-Golay"
        });
            denoisingComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            denoisingComboBox.SelectedIndex = 0;

            // ---------------------------------------------------------------------------
            // Layout and positioning
            // ---------------------------------------------------------------------------

            // =============== FIRST ROW: Generate Mesh Buttons ===============
            buttonGenerateRectangular.Location = new Point(12, 12);
            buttonGenerateRectangular.Size = new Size(180, 30);
            buttonGenerateRectangular.Text = "Generate Rectangular Mesh";

            buttonGenerateCircular.Location = new Point(12, buttonGenerateRectangular.Bottom + 8);
            buttonGenerateCircular.Size = new Size(180, 30);
            buttonGenerateCircular.Text = "Generate Circular Mesh";

            // =============== SIMULATION / DENOISING combos next to it ===============
            simulationLabel.Location = new Point(buttonGenerateRectangular.Right + 20, 12);
            simulationLabel.Size = new Size(120, 25);
            simulationLabel.Text = "Simulation Type:";
            simulationLabel.TextAlign = ContentAlignment.MiddleLeft;

            simulationComboBox.Location = new Point(simulationLabel.Left, simulationLabel.Bottom + 4);
            simulationComboBox.Size = new Size(150, 25);

            denoisingLabel.Location = new Point(simulationLabel.Left, simulationComboBox.Bottom + 8);
            denoisingLabel.Size = new Size(120, 25);
            denoisingLabel.Text = "Denoising Option:";
            denoisingLabel.TextAlign = ContentAlignment.MiddleLeft;

            denoisingComboBox.Location = new Point(denoisingLabel.Left, denoisingLabel.Bottom + 4);
            denoisingComboBox.Size = new Size(150, 25);

            // =============== DENSITY / REG next ===============
            densityLabel.Location = new Point(denoisingComboBox.Right + 20, 12);
            densityLabel.Size = new Size(80, 23);
            densityLabel.Text = "Density:";
            densityLabel.TextAlign = ContentAlignment.MiddleLeft;

            densityTextBox.Location = new Point(densityLabel.Left, densityLabel.Bottom + 4);
            densityTextBox.Size = new Size(60, 23);
            densityTextBox.Text = "1.0";
            densityTextBox.TextAlign = HorizontalAlignment.Center;

            regularizationLabel.Location = new Point(densityLabel.Left, densityTextBox.Bottom + 8);
            regularizationLabel.Size = new Size(100, 23);
            regularizationLabel.Text = "Regularization:";
            regularizationLabel.TextAlign = ContentAlignment.MiddleLeft;

            regularizationTextBox.Location = new Point(regularizationLabel.Left, regularizationLabel.Bottom + 4);
            regularizationTextBox.Size = new Size(60, 23);
            regularizationTextBox.Text = "0.01";
            regularizationTextBox.TextAlign = HorizontalAlignment.Center;

            // =============== Solver Label and Combo ===============
            solverLabel.Location = new Point(regularizationTextBox.Right + 20, 12);
            solverLabel.Size = new Size(120, 25);
            solverLabel.Text = "Select Solver:";
            solverLabel.TextAlign = ContentAlignment.MiddleLeft;

            solverComboBox.Location = new Point(solverLabel.Left, solverLabel.Bottom + 4);
            solverComboBox.Size = new Size(200, 25);

            // =============== Forward Solve Button ===============
            buttonForwardSolve.Location = new Point(1100, 12);
            buttonForwardSolve.Size = new Size(100, 30);
            buttonForwardSolve.Text = "Forward Solve";

            // =============== Inverse Solve Button ===============
            buttonInverseSolve.Location = new Point(1100, buttonForwardSolve.Bottom + 8);
            buttonInverseSolve.Size = new Size(100, 30);
            buttonInverseSolve.Text = "Inverse Solve";

            // =============== Two PictureBoxes side by side ===============
            pictureBoxPotential.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left;
            pictureBoxPotential.BackColor = Color.White;
            pictureBoxPotential.Location = new Point(12, regularizationTextBox.Bottom + 30);
            pictureBoxPotential.Size = new Size(580, 630);
            pictureBoxPotential.Name = "pictureBoxPotential";

            pictureBoxSigma.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Right;
            pictureBoxSigma.BackColor = Color.White;
            pictureBoxSigma.Location = new Point(pictureBoxPotential.Right + 8, regularizationTextBox.Bottom + 30);
            pictureBoxSigma.Size = new Size(580, 630);
            pictureBoxSigma.Name = "pictureBoxSigma";

            // =============== Form ===============
            this.ClientSize = new Size(1220, 800);
            this.Controls.Add(buttonGenerateRectangular);
            this.Controls.Add(buttonGenerateCircular);
            this.Controls.Add(simulationLabel);
            this.Controls.Add(simulationComboBox);
            this.Controls.Add(denoisingLabel);
            this.Controls.Add(denoisingComboBox);
            this.Controls.Add(densityLabel);
            this.Controls.Add(densityTextBox);
            this.Controls.Add(regularizationLabel);
            this.Controls.Add(regularizationTextBox);
            this.Controls.Add(solverLabel);
            this.Controls.Add(solverComboBox);
            this.Controls.Add(buttonForwardSolve);
            this.Controls.Add(buttonInverseSolve);
            this.Controls.Add(pictureBoxPotential);
            this.Controls.Add(pictureBoxSigma);

            this.Text = "EIT Mesh Visualizer: Potentials & Sigma";

            // done
        }
    }

}
