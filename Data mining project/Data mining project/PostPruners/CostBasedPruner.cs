﻿#region SharpLearningNameSpaces
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

#region DataMiningNameSpaces
using Data_mining_project.Metrics;
using Data_mining_project.ModelInterfaces;
#endregion DataMiningNameSpaces

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// A reduced error pruner that uses a cost dictionary with the cost of a false positive and 
    /// false negative for every class to evaluate the error.
    /// </summary>
    public sealed class CostBasedPruner : ReducedErrorPrunerBase
    {
        /// <summary>
        /// Dictionary with the following format: (class, (cost of false positive, cost of false negative)
        /// </summary>
        public readonly Dictionary<double, (double, double)> costs;

        
        /// <summary>
        /// Create the cost based pruner.
        /// </summary>
        /// <param name="costs">Cost dictionary</param>
        public CostBasedPruner(Dictionary<double, (double, double)> costs) {
            this.costs = costs;
        }

        public sealed override void Prune(IClassificationModel c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if(m.Tree.TargetNames.Any(k => !this.costs.TryGetValue(k, out _)))
            {
                throw new InvalidOperationException($"{nameof(this.costs)} does not contain all target values of the tree, which is required for the {nameof(CostBasedPruner)}");
            }
            
            base.Prune(c);
        }

        protected override double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet)
        {
            double[] prunePredictions = m.Predict(pruneSet.Observations);
            var costMetric = new CostBasedMetric(this.costs);
            return costMetric.Error(pruneSet.Targets, prunePredictions);
        }
    }
}
