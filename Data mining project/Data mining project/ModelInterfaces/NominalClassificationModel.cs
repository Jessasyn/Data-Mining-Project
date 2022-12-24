using Data_mining_project.Metrics;
using Data_mining_project.PostPruners;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.ModelInterfaces
{
    /// <summary>
    /// Interface for classification model
    /// </summary>
    public class NominalClassificationModel : ModelInterfaceBase
    {
        /// <summary>
        /// Classification metric, used in function Error, total classification error by default.
        /// </summary>
        public IClassificationMetric<double> Metric = new TotalErrorClassificationMetric<double>();
        public NominalClassificationModel(string parserPath, string targetColumn, IPruner? postPruner = null) : base(parserPath, targetColumn, postPruner) { }

        /// <summary>
        /// Calculate the classification error using this.Metric
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        public override void Error()
        {
            if (Model is null || TestSet is null)
            {
                throw new InvalidOperationException($"Cannot call {Error} before {Learn} has been called!");
            }

            double[] testPredictions = Model.Predict(TestSet.Observations);
            // In most cases, use the metric that was set.
            TestError = Metric.Error(TestSet.Targets, testPredictions);

            VariableImportance = Model.GetVariableImportance(parser.EnumerateRows(c => c != targetColumn)
                                                                                   .First().ColumnNameToIndex);
        }
        /// <summary>
        /// Calculate the classification error using CostBasedMetric with <paramref name="costs"/>
        /// </summary>
        /// <param name="costs"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public void CostError(Dictionary<double, (double, double)> costs)
        {
            if (Model is null || TestSet is null)
            {
                throw new InvalidOperationException($"Cannot call {Error} before {Learn} has been called!");
            }

            double[] testPredictions = Model.Predict(TestSet.Observations);

            // In the special case that we use cost based pruning, we take into account the costs for the error calculus.
            var costMetric = new CostBasedMetric();
            TestError = costMetric.Error(TestSet.Targets, testPredictions, costs);

            VariableImportance = Model.GetVariableImportance(parser.EnumerateRows(c => c != targetColumn)
                                                                                   .First().ColumnNameToIndex);
        }
    }
}
