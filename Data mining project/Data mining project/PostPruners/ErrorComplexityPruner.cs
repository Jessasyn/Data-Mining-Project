using Data_mining_project.Extensions;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    public sealed class ErrorComplexityPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {
            throw new NotImplementedException();
        }

        //TODO: error cost of a subtree
        public double TreeErrorCost(BinaryTree t, ObservationTargetSet trainSet, Node subRoot)
        {
            // sum of all nodes in the subtree.
            return 0d;
        }
        
        /// <summary>
        /// Calculates the error cost of a single <see cref="Node"/>.
        /// </summary>
        /// <param name="t">The tree in which the <paramref name="node"/> is located.</param>
        /// <param name="trainSet">The <see cref="ObservationTargetSet"/> that contains the training data.</param>
        /// <param name="node">The <see cref="Node"/> for which the error cost will be calculated.</param>
        /// <returns>A double, which indicates the error cost of this node.</returns>
        public double NodeErrorCost(BinaryTree t, ObservationTargetSet trainSet, Node node)
        {
            int totalExamples = trainSet.Targets.Length;

            F64Matrix populations = t.Populations(trainSet);
            double[] nodeClasses = populations.Row(node.NodeIndex);

            double misclassificationcount = nodeClasses.Sum() - nodeClasses.Where(v => v.Equals(node.Value)).Count();
            
            return misclassificationcount / totalExamples;
        }
    }
}
