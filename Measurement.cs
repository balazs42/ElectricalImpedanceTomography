using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace EIT_SOLVER
{
    public class Measurement
    {
        private double[,] MeasurementValues { get; set; }
        private int[,] MeasurementPattern { get; set; }
        private int NumElectrodes { get; set; }
        public Tuple<double[,], int[,]> Measurements { get; set; }
        public int currentMeasurement { get; set; } = 0;    // Indicates which measurement is currently being processed
        public enum DenoisingMethod
        {
            MovingAverage,
            Median,
            SavitzkyGolay,
            Gaussian, 
            Kalman, 
            Wavelet,
            Wiener,
            None
        }
        public Measurement(double[,] measurementValues, int[,] pattern, int numElectrodes)
        {
            MeasurementValues = measurementValues;
            MeasurementPattern = pattern;
            NumElectrodes = numElectrodes;

            if (MeasurementValues == null || MeasurementPattern == null)
                throw new InvalidDataException("MeasurementValues or MeasurementPattern was null during measurement initialization, check code!");

            // Creating measurements container, where the pattern matrix's rows define which electrodes are used for injection, and 
            // Values matrix represents the measured boundary voltages
            Measurements = new Tuple<double[,], int[,]>(MeasurementValues, MeasurementPattern);
        }

        public Measurement(int numPoints)
        {
            MeasurementValues = new double[numPoints, numPoints];
            MeasurementPattern = new int[numPoints, numPoints];
            NumElectrodes = numPoints;
            Measurements = new Tuple<double[,], int[,]>(MeasurementValues, MeasurementPattern);
        }

        // Returns the next measurements in the sequence
        public double[] GetNextMeasurements()
        {
            // Store how many measurement points we have
            int numMeasurements = MeasurementValues.Length;

            // Store them in a corresponding array
            double[] currentMeasurements = new double[numMeasurements];

            // Copy the data from the current measurement to the array
            for (int i = 0; i < numMeasurements; i++)
                currentMeasurements[i] = MeasurementValues[currentMeasurement, i];

            // Increment the current measurement index to keep track of which measurement we are currently processing
            currentMeasurement++;

            return currentMeasurements;
        }

        // Generate measurement data, from which the potential distribution on the boundary is symmetric
        public double[] SimulateEquvipotentialSurface(int numPoints, double equviPotential)
        {
            double[] measurements = new double[numPoints];

            for (int i = 0; i < numPoints; i++)
                measurements[i] = equviPotential;

            return measurements;
        }

        // Generate measurement data, from which the potential distribution on the boundary is gaussian
        public double[] SimulateGaussian(int numPoints, double maxPotential)
        {
            double[] measurements = new double[numPoints];

            // Parameters for the Gaussian distribution
            double mean = numPoints / 2.0; // Centered at the middle of the array
            double stdDev = numPoints / 6.0; // Standard deviation (controls spread, ~99.7% within ±3σ)

            // Calculate measurements based on the Gaussian function
            for (int i = 0; i < numPoints; i++)
            {
                double exponent = -Math.Pow(i - mean, 2) / (2 * Math.Pow(stdDev, 2));
                measurements[i] = maxPotential * Math.Exp(exponent);
            }

            return measurements;
        }

        // Add some noise to the measurements data
        public double[] AddNoiseToMeasurement(double[] measurementsToNoise, double noiseRange)
        {
            int numPoints = measurementsToNoise.Length;

            double[] measurements = new double[numPoints];

            Random random = new Random();

            // Random.NextDouble() returns a random number between 0.0 and 1.0, and we scale it with the potential range
            for (int i = 0; i < numPoints; i++)
                measurements[i] = random.NextDouble() * noiseRange;

            return measurements;
        }

        // Generate measurement data, which is noisy on the surface of the boundary
        public double[] SimulateNoise(int numPoints, double minPotential, double maxPotential)
        {
            double[] measurements = new double[numPoints];
            double range = (maxPotential - minPotential) + minPotential;

            return AddNoiseToMeasurement(measurements, range);
        }

        // Simulate assymetric potential distribution on the boundary
        public double[] SimulateAssymetric(int numPoints, double maxPotential)
        {
            int half = numPoints / 2;
            double[] measurements = new double[numPoints];

            for(int i = 0; i < numPoints; i++)
            {
                if(i < half)
                    measurements[i] = maxPotential;
                else
                    measurements[i] = 0;
            }
            return measurements;
        }

        // Denoise the measurements data
        public double[] DenoiseMeasurement(double[] measurementToDenoise, DenoisingMethod method, int windowSize = 3, double sigma = 1.0)
        {
            switch (method)
            {
                case DenoisingMethod.MovingAverage:
                    return ApplyMovingAverage(measurementToDenoise, windowSize);
                case DenoisingMethod.Median:
                    return ApplyMedianFilter(measurementToDenoise, windowSize);
                case DenoisingMethod.SavitzkyGolay:
                    return ApplySavitzkyGolay(measurementToDenoise, windowSize);
                case DenoisingMethod.Gaussian:
                    return ApplyGaussianFilter(measurementToDenoise, windowSize, sigma);
                case DenoisingMethod.Kalman:
                    return ApplyKalmanFilter(measurementToDenoise);
                case DenoisingMethod.Wavelet:
                    return ApplyWaveletFilter(measurementToDenoise);
                case DenoisingMethod.Wiener:
                    return ApplyWienerFilter(measurementToDenoise, windowSize);
                case DenoisingMethod.None:
                    return measurementToDenoise;
                default:
                    return measurementToDenoise;
            }
        }

        // Moving Average Denoisiing
        private double[] ApplyMovingAverage(double[] data, int windowSize)
        {
            double[] filtered = new double[data.Length];
            int halfWindow = windowSize / 2;

            for (int i = 0; i < data.Length; i++)
            {
                double sum = 0;
                int count = 0;

                for (int j = i - halfWindow; j <= i + halfWindow; j++)
                {
                    if (j >= 0 && j < data.Length)
                    {
                        sum += data[j];
                        count++;
                    }
                }
                filtered[i] = sum / count;
            }

            return filtered;
        }

        // -- Median Filter --
        private double[] ApplyMedianFilter(double[] data, int windowSize)
        {
            double[] filtered = new double[data.Length];
            int halfWindow = windowSize / 2;

            for (int i = 0; i < data.Length; i++)
            {
                List<double> windowValues = new List<double>();
                for (int j = i - halfWindow; j <= i + halfWindow; j++)
                {
                    if (j >= 0 && j < data.Length)
                    {
                        windowValues.Add(data[j]);
                    }
                }
                windowValues.Sort();
                int midIndex = windowValues.Count / 2;
                if (windowValues.Count % 2 == 1)
                    filtered[i] = windowValues[midIndex];
                else
                    filtered[i] = (windowValues[midIndex - 1] + windowValues[midIndex]) / 2.0;
            }

            return filtered;
        }

        // -- Savitzky-Golay Filter (Simplified) --
        private double[] ApplySavitzkyGolay(double[] data, int windowSize)
        {
            double[] filtered = new double[data.Length];
            int halfWindow = windowSize / 2;

            for (int i = 0; i < data.Length; i++)
            {
                // Collect points in [i-halfWindow, i+halfWindow]
                List<double> xVals = new List<double>();
                List<double> yVals = new List<double>();
                for (int j = i - halfWindow; j <= i + halfWindow; j++)
                {
                    if (j >= 0 && j < data.Length)
                    {
                        xVals.Add(j);
                        yVals.Add(data[j]);
                    }
                }

                // Fit a second-order polynomial to these points: y = a + b*x + c*x^2
                double[] coeffs = FitPolynomialOrder2(xVals, yVals);

                // Evaluate the polynomial at x = i
                double denoisedVal = coeffs[0] + coeffs[1] * i + coeffs[2] * i * i;
                filtered[i] = denoisedVal;
            }

            return filtered;
        }

        // Gaussian Filter 
        private double[] ApplyGaussianFilter(double[] data, int windowSize, double sigma)
        {
            // Build a Gaussian kernel
            double[] kernel = BuildGaussianKernel(windowSize, sigma);
            double[] filtered = new double[data.Length];
            int halfWindow = windowSize / 2;

            for (int i = 0; i < data.Length; i++)
            {
                double sum = 0.0;
                double weightSum = 0.0;
                for (int j = -halfWindow; j <= halfWindow; j++)
                {
                    int index = i + j;
                    if (index >= 0 && index < data.Length)
                    {
                        double w = kernel[j + halfWindow];
                        sum += w * data[index];
                        weightSum += w;
                    }
                }
                filtered[i] = sum / weightSum;
            }

            return filtered;
        }

        // Build Kernel for denoising
        private double[] BuildGaussianKernel(int windowSize, double sigma)
        {
            double[] kernel = new double[windowSize];
            int halfWindow = windowSize / 2;
            double sigma2 = 2 * sigma * sigma;
            double norm = 1.0 / (Math.Sqrt(2 * Math.PI) * sigma);

            for (int i = -halfWindow; i <= halfWindow; i++)
            {
                double x = i * i;
                kernel[i + halfWindow] = norm * Math.Exp(-x / sigma2);
            }
            return kernel;
        }

        // Kalman Filter (Very basic 1D example)
        private double[] ApplyKalmanFilter(double[] data)
        {
            double[] filtered = new double[data.Length];

            // Initial guesses
            double estimate = data[0];
            double errorEstimate = 1.0;
            double errorMeasure = 1.0; // measurement noise
            double processNoise = 0.01; // process noise

            filtered[0] = estimate;

            for (int i = 1; i < data.Length; i++)
            {
                // Prediction update
                errorEstimate += processNoise;

                // Measurement update
                double kalmanGain = errorEstimate / (errorEstimate + errorMeasure);
                estimate = estimate + kalmanGain * (data[i] - estimate);
                errorEstimate = (1.0 - kalmanGain) * errorEstimate;

                filtered[i] = estimate;
            }

            return filtered;
        }

        // -- Wavelet Filter (Simplified) --
        // A typical approach: (1) Decompose signal, (2) threshold wavelet coefficients, (3) reconstruct.
        // This is a naive example using a simple Haar transform.
        private double[] ApplyWaveletFilter(double[] data)
        {
            // Step 1: Discrete Wavelet Transform (1 level)
            int n = data.Length;
            if (n < 2) return data;

            // Haar wavelet transform (single-level)
            int half = n / 2;
            double[] approx = new double[half];
            double[] detail = new double[half];

            for (int i = 0; i < half; i++)
            {
                approx[i] = (data[2 * i] + data[2 * i + 1]) / Math.Sqrt(2);
                detail[i] = (data[2 * i] - data[2 * i + 1]) / Math.Sqrt(2);
            }

            // Step 2: Threshold detail coefficients
            double threshold = ComputeWaveletThreshold(detail);
            for (int i = 0; i < half; i++)
            {
                // Simple hard thresholding
                if (Math.Abs(detail[i]) < threshold)
                {
                    detail[i] = 0.0;
                }
            }

            // Step 3: Inverse wavelet transform (1 level)
            double[] reconstructed = new double[n];
            for (int i = 0; i < half; i++)
            {
                reconstructed[2 * i] = (approx[i] + detail[i]) / Math.Sqrt(2);
                reconstructed[2 * i + 1] = (approx[i] - detail[i]) / Math.Sqrt(2);
            }

            return reconstructed;
        }

        private double ComputeWaveletThreshold(double[] detail)
        {
            // A simple approach: threshold = k * (median of abs detail) / 0.6745
            // k is usually ~1 to 3. We'll pick k=1 here.
            double[] absVals = detail.Select(x => Math.Abs(x)).ToArray();
            Array.Sort(absVals);
            double median = absVals[absVals.Length / 2];
            double threshold = median / 0.6745; // k=1
            return threshold;
        }

        // Wiener Filter (Local approach) 
        // This is a simplified version that approximates the local mean and variance in a window and applies Wiener filter.
        private double[] ApplyWienerFilter(double[] data, int windowSize)
        {
            double[] filtered = new double[data.Length];
            int halfWindow = windowSize / 2;

            double globalVariance = ComputeVariance(data);

            for (int i = 0; i < data.Length; i++)
            {
                double localMean = 0.0;
                double localVar = 0.0;
                int count = 0;

                // Compute local mean
                for (int j = i - halfWindow; j <= i + halfWindow; j++)
                {
                    if (j >= 0 && j < data.Length)
                    {
                        localMean += data[j];
                        count++;
                    }
                }

                localMean /= (count == 0 ? 1 : count);

                // Compute local variance
                for (int j = i - halfWindow; j <= i + halfWindow; j++)
                {
                    if (j >= 0 && j < data.Length)
                    {
                        double diff = data[j] - localMean;
                        localVar += diff * diff;
                    }
                }
                localVar /= (count == 0 ? 1 : count);

                // Wiener filter formula:
                // filtered_value = mean + (max(0, localVar - noiseVar) / localVar) * (original - mean)
                // We'll assume noiseVar = globalVariance for simplicity, or localVar if smaller.
                // In practice, you might estimate noise variance differently.

                double noiseVar = globalVariance; // naive assumption
                double pixelVal = data[i];

                double ratio = 0.0;
                if (localVar > 0)
                {
                    ratio = Math.Max(0, localVar - noiseVar) / localVar;
                }

                filtered[i] = localMean + ratio * (pixelVal - localMean);
            }

            return filtered;
        }

        // Computes the variance of the data
        private double ComputeVariance(double[] data)
        {
            double mean = data.Average();
            double variance = 0.0;
            foreach (var val in data)
            {
                double diff = val - mean;
                variance += diff * diff;
            }
            variance /= data.Length;
            return variance;
        }

        // Helper method to fit a second-order polynomial using a least squares approach (Savitzky-Golay) 
        private double[] FitPolynomialOrder2(List<double> xVals, List<double> yVals)
        {
            double n = xVals.Count;
            double sumX = 0;
            double sumX2 = 0;
            double sumX3 = 0;
            double sumX4 = 0;
            double sumY = 0;
            double sumXY = 0;
            double sumX2Y = 0;

            for (int i = 0; i < xVals.Count; i++)
            {
                double x = xVals[i];
                double y = yVals[i];
                sumX += x;
                sumX2 += x * x;
                sumX3 += x * x * x;
                sumX4 += x * x * x * x;
                sumY += y;
                sumXY += x * y;
                sumX2Y += x * x * y;
            }

            double[,] A = new double[3, 3] {
            { n,     sumX,  sumX2 },
            { sumX,  sumX2, sumX3 },
            { sumX2, sumX3, sumX4 }
        };

            double[] B = new double[3] { sumY, sumXY, sumX2Y };

            double[] solution = Solve3x3(A, B);
            return solution; // [a, b, c]
        }

        // Solve 3x3 system of equations
        private double[] Solve3x3(double[,] A, double[] B)
        {
            double detA = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
                        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
                        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]);

            if (Math.Abs(detA) < 1e-12)
            {
                // Degenerate, just return zeros or throw an exception
                return new double[3];
            }

            double[] result = new double[3];

            // Use Cramer's rule
            for (int i = 0; i < 3; i++)
            {
                double[,] temp = (double[,])A.Clone();
                temp[0, i] = B[0];
                temp[1, i] = B[1];
                temp[2, i] = B[2];

                double detTemp = temp[0, 0] * (temp[1, 1] * temp[2, 2] - temp[1, 2] * temp[2, 1])
                               - temp[0, 1] * (temp[1, 0] * temp[2, 2] - temp[1, 2] * temp[2, 0])
                               + temp[0, 2] * (temp[1, 0] * temp[2, 1] - temp[1, 1] * temp[2, 0]);

                result[i] = detTemp / detA;
            }

            return result;
        }


        // Interpolage boundary data equally
        private double[] InterpolateBoundaryData(Mesh Mesh, double[] BoundaryVoltages, int numPoints)
        {
            int numMeasurements = BoundaryVoltages.Length;
            int numBoundaryVertices = numPoints;

            // Validate inputs
            if (numMeasurements > numBoundaryVertices)
            {
                throw new ArgumentException("Number of measurements cannot exceed the number of boundary vertices.");
            }

            // Initialize the result array with default potentials (e.g., zeros)
            double[] interpolatedPotentials = new double[numBoundaryVertices];

            // If the mesh is circular, proceed with circular interpolation
            if (Mesh.IsCircular)
            {
                // Calculate the step size to distribute measurements equally
                double step = (double)numBoundaryVertices / numMeasurements;
                List<int> measurementIndices = new List<int>();

                // Assign measurements at equally spaced indices
                for (int i = 0; i < numMeasurements; i++)
                {
                    // Calculate the exact position for the measurement
                    double position = i * step;

                    // Find the closest integer index
                    int index = (int)Math.Round(position) % numBoundaryVertices;

                    // Ensure no duplicate indices by adjusting if necessary
                    while (measurementIndices.Contains(index))
                    {
                        index = (index + 1) % numBoundaryVertices;
                    }

                    // Add to the list of measurement indices
                    measurementIndices.Add(index);

                    // Assign the measured potential
                    interpolatedPotentials[index] = BoundaryVoltages[i];
                }

                // Interpolate potentials between measurements
                for (int i = 0; i < numMeasurements; i++)
                {
                    // Current and next measurement indices
                    int currentIndex = measurementIndices[i];
                    int nextIndex = measurementIndices[(i + 1) % numMeasurements];

                    // Current and next potentials
                    double currentPotential = interpolatedPotentials[currentIndex];
                    double nextPotential = interpolatedPotentials[nextIndex];

                    // Calculate the number of vertices between current and next measurements
                    int count;
                    if (nextIndex > currentIndex)
                    {
                        count = nextIndex - currentIndex - 1;
                    }
                    else
                    {
                        count = numBoundaryVertices - currentIndex + nextIndex - 1;
                    }

                    // Interpolate potentials for the vertices between current and next measurements
                    for (int j = 1; j <= count; j++)
                    {
                        int interpolatedIndex = (currentIndex + j) % numBoundaryVertices;
                        double fraction = (double)j / (count + 1); // Fraction between 0 and 1
                        double interpolatedPotential = currentPotential + fraction * (nextPotential - currentPotential);

                        // Assign the interpolated potential
                        interpolatedPotentials[interpolatedIndex] = interpolatedPotential;
                    }
                }
            }
            else
            {
                // Handle non-circular meshes if necessary
                // For simplicity, we'll implement linear interpolation assuming the boundary is a polygon

                // Distribute measurements equally along the boundary
                double step = (double)numBoundaryVertices / numMeasurements;
                List<int> measurementIndices = new List<int>();

                for (int i = 0; i < numMeasurements; i++)
                {
                    double position = i * step;
                    int index = (int)Math.Round(position) % numBoundaryVertices;

                    // Avoid duplicate indices
                    while (measurementIndices.Contains(index))
                    {
                        index = (index + 1) % numBoundaryVertices;
                    }

                    measurementIndices.Add(index);
                    interpolatedPotentials[index] = BoundaryVoltages[i];
                }

                // Interpolate potentials between measurements
                for (int i = 0; i < numMeasurements; i++)
                {
                    int currentIndex = measurementIndices[i];
                    int nextIndex = measurementIndices[(i + 1) % numMeasurements];

                    double currentPotential = interpolatedPotentials[currentIndex];
                    double nextPotential = interpolatedPotentials[nextIndex];

                    int count;
                    if (nextIndex > currentIndex)
                    {
                        count = nextIndex - currentIndex - 1;
                    }
                    else
                    {
                        count = numBoundaryVertices - currentIndex + nextIndex - 1;
                    }

                    for (int j = 1; j <= count; j++)
                    {
                        int interpolatedIndex = (currentIndex + j) % numBoundaryVertices;
                        double fraction = (double)j / (count + 1);
                        double interpolatedPotential = currentPotential + fraction * (nextPotential - currentPotential);

                        interpolatedPotentials[interpolatedIndex] = interpolatedPotential;
                    }
                }
            }

            return interpolatedPotentials;
        }

    }
}
