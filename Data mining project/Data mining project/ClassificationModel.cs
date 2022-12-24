using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project
{
    public class ClassificationModel : ModelInterfaceBase
    {
        public void Error()
        {
            if (this.Model is null || this.TestSet is null)
            {
                throw new InvalidOperationException($"Cannot call {this.Error} before {this.Learn} has been called!");
            }

            double[] testPredictions = this.Model.Predict(this.TestSet.Observations);

            MeanSquaredErrorRegressionMetric metric = new MeanSquaredErrorRegressionMetric();
            this.TestError = metric.Error(this.TestSet.Targets, testPredictions);

            this.VariableImportance = this.Model.GetVariableImportance(this._parser.EnumerateRows(c => c != this._targetColumn)
                                                                                   .First().ColumnNameToIndex);
        }
    }
}
