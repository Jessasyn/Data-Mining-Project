using Data_mining_project.Extensions;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Metrics.Regression;

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// An implementation of <see cref="IPruner"/> which uses the reduced error pruning algorithm to prune a decision tree.
    /// </summary>
    public sealed class ReducedErrorPruner : PrunerBase
    {
        /// <summary>
        /// The metric that is used to measure the error of the tree in <see cref="PruneSetError(ClassificationDecisionTreeModel, ObservationTargetSet)"/>.
        /// </summary>
        private readonly MeanSquaredErrorRegressionMetric _metric = new MeanSquaredErrorRegressionMetric();
        
        /// <summary>
        /// Prunes the <paramref name="c"/> using the reduced error pruning algorithm.
        /// </summary>
        /// <param name="c"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public override void Prune(IClassifier c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if (c.GetTrainingSet() is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training data set, which is required for reduced error pruning!");
            }

            if (c.GetPruneSet() is not ObservationTargetSet pruneSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a pruning data set, which is required for reduced error pruning!");
            }

            BinaryTree t = m.Tree;

            // Matrix that stores the population of the classes at each node.
            F64Matrix populations = t.Populations(trainSet);

            for (int i = t.Nodes.Count - 1; i >= 0; i--)
            {
                Node oldNode = t.Nodes[i];

                // Only proceed if this is a non-leaf node
                if (oldNode.LeftIndex != -1 && oldNode.RightIndex != -1)
                {
                    double prePrunedError = this.PruneSetError(m, pruneSet);

                    // Find the most frequent class of this node using the populations matrix
                    double mostFrequentClass = t.MostFrequentClass(i, populations);

                    // Create a new node which is basically a copy of the old node but without childeren.
                    this.PruneNode(i, mostFrequentClass, t);
                    
                    // Now if the accuracy has stayed the same or has improved, keep the change. Otherwise, we put back the old node.
                    if (this.PruneSetError(m, pruneSet) > prePrunedError)
                    {
                        t.Nodes[i] = oldNode; // Revert change
                    }
                }
            }
        }
        
        /// <summary>
        /// Determines the accuracy of <see cref="ClassificationDecisionTreeModel"/> <paramref name="m"/> 
        /// on the <see cref="ObservationTargetSet"/> <paramref name="pruneSet"/>, and returns that as a <see cref="double"/>.
        /// </summary>
        /// <returns>A <see cref="double"/>, which is the error rate that is computed.</returns>
        private double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet)
        {
            double[] prunePredictions = m.Predict(pruneSet.Observations);

            return this._metric.Error(pruneSet.Targets, prunePredictions);
        }
    }
}
