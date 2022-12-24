using Data_mining_project.PostPruners;
using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.ModelInterfaces
{
    /// <summary>
    /// Interface for ordinal classification model
    /// </summary>
    internal class OrdinalClassificationModel : ClassificationModelBase
    {
        /// <summary>
        /// Regression metric, used in function Error, mean squared error by default.
        /// </summary>
        public IRegressionMetric Metric = new MeanSquaredErrorRegressionMetric();
        public OrdinalClassificationModel(string parserPath, string targetColumn, IPruner? postPruner = null) : base(parserPath, targetColumn, postPruner) { }
        /// <summary>
        /// Calculate the regression error using this.Metric
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        public override void Error()
        {
            if (Model is null || TestSet is null)
            {
                throw new InvalidOperationException($"Cannot call {Error} before {Learn} has been called!");
            }

            double[] testPredictions = Model.Predict(TestSet.Observations);

            MeanSquaredErrorRegressionMetric metric = new MeanSquaredErrorRegressionMetric();
            TestError = metric.Error(TestSet.Targets, testPredictions);

            VariableImportance = Model.GetVariableImportance(parser.EnumerateRows(c => c != targetColumn)
                                                                                   .First().ColumnNameToIndex);
        }
    }
}
