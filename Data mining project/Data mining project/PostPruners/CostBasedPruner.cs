#region SharpLearningNameSpaces
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

#region DataMiningNameSpaces
using Data_mining_project.Metrics;
#endregion DataMiningNameSpaces

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// A reduced error pruner that uses a cost dictionary with the cost of a false positive and 
    /// false negative for every class to evaluate the error.
    /// </summary>
    public sealed class CostBasedPruner : ReducedErrorPruner
    {
        /// <summary>
        /// Dictionary with the following format: (class, (cost of false positive, cost of false negative)
        /// </summary>
        private readonly Dictionary<double, (double, double)> _costs;

        /// <summary>
        /// Metric used for evaluating accuracy with the pruning set.
        /// </summary>
        private readonly CostBasedMetric _metric = new CostBasedMetric();
        
        /// <summary>
        /// Create the cost based pruner.
        /// </summary>
        /// <param name="costs">Cost dictionary</param>
        public CostBasedPruner(Dictionary<double, (double, double)> costs) {
            this._costs = costs;
        }

        public sealed override void Prune(IClassifier c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if(m.Tree.TargetNames.Any(k => !this._costs.TryGetValue(k, out _)))
            {
                throw new InvalidOperationException($"{nameof(this._costs)} does not contain all target values of the tree, which is required for the {nameof(CostBasedPruner)}");
            }
            
            base.Prune(c);
        }
        
        protected sealed override double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet)
        {
            double[] prunePredictions = m.Predict(pruneSet.Observations);
            return this._metric.Error(prunePredictions, pruneSet.Targets, this._costs);
        }
    }
}
