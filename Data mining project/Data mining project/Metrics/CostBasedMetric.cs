using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.Metrics
{
    public class CostBasedMetric
    {
        /// <summary>
        /// Calculate error using costs.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="predicted"></param>
        /// <param name="costs">Dictionary with as key the class and a tuple (cost of false positive, cost of false negative)</param>
        /// <returns>Error rate</returns>
        /// <exception cref="NotImplementedException"></exception>
        public double Error(double[] targets, double[] predictions, Dictionary<double, (double, double)> costs)
        {
            if (targets.Length != predictions.Length)
            {
                throw new ArgumentException("targets and predictions length do not match");
            }

            double meanSquareError = 0d;
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predictions[i];
                double error = 0d;
                if (targetValue != estimate)
                {

                }
                meanSquareError += error;
            }
            meanSquareError *= (1.0 / targets.Length);

            return meanSquareError;
        }
    }
}
