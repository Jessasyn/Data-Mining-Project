using Data_mining_project.Metrics;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.PostPruners
{
    public class ReducedErrorPruner : ReducedErrorPrunerBase
    {
        /// <summary>
        /// Metric used for pruning, MeanSquaredErrorRegressionMetric by default.
        /// </summary>
        public IRegressionMetric metric = new MeanSquaredErrorRegressionMetric();
        protected override double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet)
        {
            double[] prunePredictions = m.Predict(pruneSet.Observations);

            return this.metric.Error(pruneSet.Targets, prunePredictions);
        }
    }
}
