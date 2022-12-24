using SharpLearning.Metrics.Classification;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.Metrics
{
    public class CostBasedMetric
    {
        private Dictionary<double, (double, double)> _costs;
        public CostBasedMetric(Dictionary<double, (double, double)> costs) 
        {
            this._costs = costs;
        }
        /// <summary>
        /// Calculate error using costs.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="predicted"></param>
        /// <param name="costs">Dictionary with as key the class and a tuple (cost of false positive, cost of false negative)</param>
        /// <returns>Error rate</returns>
        /// <exception cref="NotImplementedException"></exception>
        public double Error(double[] targets, double[] predictions)
        {
            if (targets.Length != predictions.Length)
            {
                throw new ArgumentException("targets and predictions length do not match");
            }

            double weightedError = 0d;
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predictions[i];
                double error = 0d;
                if (targetValue != estimate)
                {
                    error += _costs[estimate].Item1 / 2; // False positive estimate
                    error += _costs[targetValue].Item2 / 2; // False negative targetValue
                }
                weightedError += error;
            }
            weightedError *= (1.0 / targets.Length);

            return weightedError;
        }
    }
}
