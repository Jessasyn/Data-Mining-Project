using Data_mining_project.Metrics;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Models;

namespace Data_mining_project.PostPruners
{
    internal class CostBasedPruner : ReducedErrorPruner
    {
        public Dictionary<double, (double, double)> Costs { get; set; }

        public CostBasedPruner(Dictionary<double, (double, double)> costs) {
            Costs = costs;
        }

        public override void Prune(IClassifier c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if (c.GetTrainingSet() is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training data set, which is required for the {nameof(CostBasedPruner)}!");
            }

            if (c.GetPruneSet() is not ObservationTargetSet pruneSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a pruning data set, which is required for the {nameof(CostBasedPruner)}!");
            }
            base.Prune(c);
        }
        
        public override double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet)
        {
            double[] prunePredictions = m.Predict(pruneSet.Observations);
            var metric = new CostBasedMetric();
            return metric.Error(prunePredictions, pruneSet.Targets, Costs);
        }
    }
}
