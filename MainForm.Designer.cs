// MainForm.cs
using System;
using System.Drawing;
using System.Windows.Forms;

namespace EIT_SOLVER
{
    partial class MainForm
    {
        private System.ComponentModel.IContainer components = null;
        private System.Windows.Forms.PictureBox pictureBoxMesh;
        private System.Windows.Forms.Button buttonGenerateRectangular;
        private System.Windows.Forms.Button buttonGenerateCircular;
        private System.Windows.Forms.Button buttonSolve;

        private System.Windows.Forms.TextBox densityTextBox;
        private System.Windows.Forms.Label densityLabel;

        private System.Windows.Forms.TextBox regularizationTextBox;
        private System.Windows.Forms.Label regularizationLabel;

        private System.Windows.Forms.Label solverLabel;
        private System.Windows.Forms.ComboBox solverComboBox;

        // New Controls
        private System.Windows.Forms.Label simulationLabel;
        private System.Windows.Forms.ComboBox simulationComboBox;
        private System.Windows.Forms.Label denoisingLabel;
        private System.Windows.Forms.ComboBox denoisingComboBox;

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
            pictureBoxMesh = new PictureBox();
            buttonGenerateRectangular = new Button();
            buttonGenerateCircular = new Button();

            densityLabel = new Label();
            densityTextBox = new TextBox();

            regularizationLabel = new Label();
            regularizationTextBox = new TextBox();

            solverLabel = new Label();
            solverComboBox = new ComboBox();
            solverComboBox.Items.AddRange(new object[] {
            "LU Decomposition",
            "Singular Value Decomposition",
            "Conjugate Gradient",
            "Without Lagrange Multipliers"});

            solverComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            solverComboBox.SelectedIndex = 0; // Default selection

            buttonSolve = new Button();

            // Initialize New Controls
            simulationLabel = new Label();
            simulationComboBox = new ComboBox();
            denoisingLabel = new Label();
            denoisingComboBox = new ComboBox();

            // Populate Simulation ComboBox
            simulationComboBox.Items.AddRange(new object[] {
                "Eqvipotential",
                "Assymtric",
                "Noise"});
            simulationComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            simulationComboBox.SelectedIndex = 0; // Default selection

            // Populate Denoising ComboBox
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
            denoisingComboBox.SelectedIndex = 0; // Default selection

            ((System.ComponentModel.ISupportInitialize)pictureBoxMesh).BeginInit();
            SuspendLayout();

            // 
            // buttonGenerateRectangular
            // 
            buttonGenerateRectangular.Location = new Point(12, 12);
            buttonGenerateRectangular.Name = "buttonGenerateRectangular";
            buttonGenerateRectangular.Size = new Size(180, 30);
            buttonGenerateRectangular.TabIndex = 1;
            buttonGenerateRectangular.Text = "Generate Rectangular Mesh";
            buttonGenerateRectangular.UseVisualStyleBackColor = true;

            // 
            // buttonGenerateCircular
            // 
            buttonGenerateCircular.Location = new Point(12, buttonGenerateRectangular.Bottom + 8); // Positioned below Rectangular
            buttonGenerateCircular.Name = "buttonGenerateCircular";
            buttonGenerateCircular.Size = new Size(180, 30);
            buttonGenerateCircular.TabIndex = 2;
            buttonGenerateCircular.Text = "Generate Circular Mesh";
            buttonGenerateCircular.UseVisualStyleBackColor = true;

            // 
            // simulationLabel
            // 
            simulationLabel.Location = new Point(buttonGenerateRectangular.Right + 20, 12);
            simulationLabel.Name = "simulationLabel";
            simulationLabel.Size = new Size(120, 25);
            simulationLabel.TabIndex = 10;
            simulationLabel.Text = "Simulation Type:";
            simulationLabel.TextAlign = ContentAlignment.MiddleLeft;

            // 
            // simulationComboBox
            // 
            simulationComboBox.Location = new Point(simulationLabel.Left, simulationLabel.Bottom + 4);
            simulationComboBox.Name = "simulationComboBox";
            simulationComboBox.Size = new Size(150, 25);
            simulationComboBox.TabIndex = 11;

            // 
            // denoisingLabel
            // 
            denoisingLabel.Location = new Point(simulationLabel.Left, simulationComboBox.Bottom + 8);
            denoisingLabel.Name = "denoisingLabel";
            denoisingLabel.Size = new Size(120, 25);
            denoisingLabel.TabIndex = 12;
            denoisingLabel.Text = "Denoising Option:";
            denoisingLabel.TextAlign = ContentAlignment.MiddleLeft;

            // 
            // denoisingComboBox
            // 
            denoisingComboBox.Location = new Point(denoisingLabel.Left, denoisingLabel.Bottom + 4);
            denoisingComboBox.Name = "denoisingComboBox";
            denoisingComboBox.Size = new Size(150, 25);
            denoisingComboBox.TabIndex = 13;

            // 
            // densityLabel
            // 
            densityLabel.Location = new Point(denoisingComboBox.Right + 20, 12);
            densityLabel.Name = "densityLabel";
            densityLabel.Size = new Size(80, 23);
            densityLabel.TabIndex = 3;
            densityLabel.Text = "Density:";
            densityLabel.TextAlign = ContentAlignment.MiddleLeft;

            // 
            // densityTextBox
            // 
            densityTextBox.Location = new Point(densityLabel.Left, densityLabel.Bottom + 4);
            densityTextBox.Name = "densityTextBox";
            densityTextBox.Size = new Size(60, 23);
            densityTextBox.TabIndex = 4;
            densityTextBox.Text = "1.0";
            densityTextBox.TextAlign = HorizontalAlignment.Center;

            // 
            // regularizationLabel
            // 
            regularizationLabel.Location = new Point(densityLabel.Left, densityTextBox.Bottom + 8);
            regularizationLabel.Name = "regularizationLabel";
            regularizationLabel.Size = new Size(100, 23);
            regularizationLabel.TabIndex = 6;
            regularizationLabel.Text = "Regularization:";
            regularizationLabel.TextAlign = ContentAlignment.MiddleLeft;

            // 
            // regularizationTextBox
            // 
            regularizationTextBox.Location = new Point(regularizationLabel.Left, regularizationLabel.Bottom + 4);
            regularizationTextBox.Name = "regularizationTextBox";
            regularizationTextBox.Size = new Size(60, 23);
            regularizationTextBox.TabIndex = 7;
            regularizationTextBox.Text = "0.01";
            regularizationTextBox.TextAlign = HorizontalAlignment.Center;

            // 
            // solverLabel
            // 
            solverLabel.Location = new Point(regularizationTextBox.Right + 20, 12);
            solverLabel.Name = "solverLabel";
            solverLabel.Size = new Size(120, 25); // Increased width for better readability
            solverLabel.TabIndex = 8;
            solverLabel.Text = "Select Solver:";
            solverLabel.TextAlign = ContentAlignment.MiddleLeft;

            // 
            // solverComboBox
            // 
            solverComboBox.Location = new Point(solverLabel.Left, solverLabel.Bottom + 4);
            solverComboBox.Name = "solverComboBox";
            solverComboBox.Size = new Size(200, 25); // Increased width
            solverComboBox.TabIndex = 9;

            // 
            // buttonSolve
            // 
            buttonSolve.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            buttonSolve.Location = new Point(1100, 12);
            buttonSolve.Name = "buttonSolve";
            buttonSolve.Size = new Size(80, 30);
            buttonSolve.TabIndex = 5;
            buttonSolve.Text = "Solve";
            buttonSolve.UseVisualStyleBackColor = true;

            // 
            // pictureBoxMesh
            // 
            pictureBoxMesh.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            pictureBoxMesh.BackColor = Color.White;
            pictureBoxMesh.Location = new Point(12, regularizationTextBox.Bottom + 30);
            pictureBoxMesh.Name = "pictureBoxMesh";
            pictureBoxMesh.Size = new Size(1168, 630);
            pictureBoxMesh.TabIndex = 0;
            pictureBoxMesh.TabStop = false;

            // 
            // MainForm
            // 
            ClientSize = new Size(1200, 800);
            Controls.Add(denoisingComboBox);
            Controls.Add(denoisingLabel);
            Controls.Add(simulationComboBox);
            Controls.Add(simulationLabel);
            Controls.Add(solverComboBox);
            Controls.Add(solverLabel);
            Controls.Add(regularizationTextBox);
            Controls.Add(regularizationLabel);
            Controls.Add(densityTextBox);
            Controls.Add(densityLabel);
            Controls.Add(buttonSolve);
            Controls.Add(buttonGenerateCircular);
            Controls.Add(buttonGenerateRectangular);
            Controls.Add(pictureBoxMesh);
            Name = "MainForm";
            Text = "EIT Mesh Visualizer";
            ((System.ComponentModel.ISupportInitialize)pictureBoxMesh).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }
    }
}
