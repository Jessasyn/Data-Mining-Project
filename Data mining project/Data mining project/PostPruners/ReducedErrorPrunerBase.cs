#region DataMiningNameSpaces
using Data_mining_project.Extensions;
#endregion DataMiningNameSpaces

#region SharpLearningNameSpaces
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Metrics.Regression;
using SharpLearning.Containers;
using Data_mining_project.ModelInterfaces;
#endregion SharpLearningNameSpaces

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// An implementation of <see cref="IPruner"/> which uses the reduced error pruning algorithm to prune a decision tree.
    /// </summary>
    public abstract class ReducedErrorPrunerBase : PrunerBase
    {
        /// <summary>
        /// The metric that is used to measure the error of the tree in <see cref="PruneSetError(ClassificationDecisionTreeModel, ObservationTargetSet)"/>.
        /// </summary>
        public object metric = new MeanSquaredErrorRegressionMetric();

        /// <summary>
        /// Prunes the <paramref name="c"/> using the reduced error pruning algorithm.
        /// </summary>
        /// <param name="c">The classifier to prune.</param>
        /// <exception cref="InvalidOperationException">If the state does not allow for pruning to occur.</exception>
        public override void Prune(IModelInterface c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if (c.GetTrainingSet() is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training data set, which is required for the {nameof(ReducedErrorPrunerBase)}!");
            }

            if (c.GetPruneSet() is not ObservationTargetSet pruneSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a pruning data set, which is required for the {nameof(ReducedErrorPrunerBase)}!");
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

                    // Find the most frequent class of this node using the populations matrix.
                    double mostFrequentClass = t.MostFrequentClass(i, populations);

                    // Create a new node which, which is identical to the old node, but with all of its children removed.
                    t.PruneNode(i, mostFrequentClass);
                    
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
        protected abstract double PruneSetError(ClassificationDecisionTreeModel m, ObservationTargetSet pruneSet);
    }
}
