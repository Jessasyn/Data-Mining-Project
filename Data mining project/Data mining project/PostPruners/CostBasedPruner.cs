#region SharpLearningNameSpaces
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

#region DataMiningNameSpaces
using Data_mining_project.Metrics;
#endregion DataMiningNameSpaces

namespace Data_mining_project.PostPruners
{
    public sealed class CostBasedPruner : ReducedErrorPruner
    {
        private readonly Dictionary<double, (double, double)> _costs;

        private readonly CostBasedMetric _metric = new CostBasedMetric();
        
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
        
        public sealed override double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet)
        {
            double[] prunePredictions = m.Predict(pruneSet.Observations);
            return this._metric.Error(prunePredictions, pruneSet.Targets, this._costs);
        }
    }
}
