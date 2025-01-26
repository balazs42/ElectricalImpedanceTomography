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
        
        private System.Windows.Forms.TextBox sizeTextBox;
        private System.Windows.Forms.Label sizeLabel;

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
            this.pictureBoxMesh = new System.Windows.Forms.PictureBox();
            this.buttonGenerateRectangular = new System.Windows.Forms.Button();
            this.buttonGenerateCircular = new System.Windows.Forms.Button();
            this.densityLabel = new System.Windows.Forms.Label();
            this.densityTextBox = new System.Windows.Forms.TextBox();
            this.buttonSolve = new System.Windows.Forms.Button();

            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMesh)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBoxMesh
            // 
            this.pictureBoxMesh.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                                                                             | System.Windows.Forms.AnchorStyles.Left)
                                                                            | System.Windows.Forms.AnchorStyles.Right)));
            this.pictureBoxMesh.BackColor = System.Drawing.Color.White;
            this.pictureBoxMesh.Location = new System.Drawing.Point(12, 50);
            this.pictureBoxMesh.Name = "pictureBoxMesh";
            this.pictureBoxMesh.Size = new System.Drawing.Size(760, 499);
            this.pictureBoxMesh.TabIndex = 0;
            this.pictureBoxMesh.TabStop = false;
            // 
            // buttonGenerateRectangular
            // 
            this.buttonGenerateRectangular.Location = new System.Drawing.Point(12, 12);
            this.buttonGenerateRectangular.Name = "buttonGenerateRectangular";
            this.buttonGenerateRectangular.Size = new System.Drawing.Size(150, 32);
            this.buttonGenerateRectangular.TabIndex = 1;
            this.buttonGenerateRectangular.Text = "Generate Rectangular Mesh";
            this.buttonGenerateRectangular.UseVisualStyleBackColor = true;
            // 
            // buttonGenerateCircular
            // 
            this.buttonGenerateCircular.Location = new System.Drawing.Point(168, 12);
            this.buttonGenerateCircular.Name = "buttonGenerateCircular";
            this.buttonGenerateCircular.Size = new System.Drawing.Size(150, 32);
            this.buttonGenerateCircular.TabIndex = 2;
            this.buttonGenerateCircular.Text = "Generate Circular Mesh";
            this.buttonGenerateCircular.UseVisualStyleBackColor = true;
            //
            // Density Label
            //
            this.densityLabel.Location = new System.Drawing.Point(324, 12);
            this.densityLabel.Name = "densityLabel";
            this.densityLabel.Size = new System.Drawing.Size(150, 32);
            this.densityLabel.Text = "Density:";
            // 
            // Density TextBox
            //
            this.densityTextBox.Location = new System.Drawing.Point(480, 12);
            this.densityTextBox.Name = "densityTextBox";
            this.densityTextBox.Size = new System.Drawing.Size(150, 32);
            this.densityTextBox.Text = "1,0";
            this.densityTextBox.TextAlign = HorizontalAlignment.Center;
            this.densityTextBox.Enabled = true;
            //
            // Solve Button
            //
            this.buttonSolve.Location = new System.Drawing.Point(636, 12);
            this.buttonSolve.Name = "buttonSolve";
            this.buttonSolve.Size = new System.Drawing.Size(100, 32);
            this.buttonSolve.TabIndex = 3;
            this.buttonSolve.Text = "Solve";
            this.buttonSolve.UseVisualStyleBackColor = true;
            // 
            // MainForm
            // 
            this.ClientSize = new System.Drawing.Size(784, 561);
            this.Controls.Add(this.buttonGenerateCircular);
            this.Controls.Add(this.buttonGenerateRectangular);
            this.Controls.Add(this.pictureBoxMesh);
            this.Controls.Add(this.densityLabel);
            this.Controls.Add(this.densityTextBox);
            this.Controls.Add(this.buttonSolve);
            this.Name = "MainForm";
            this.Text = "EIT Mesh Visualizer";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMesh)).EndInit();
            this.ResumeLayout(false);

        }
    }
}

