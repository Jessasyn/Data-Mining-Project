#region DataminingNameSpaces
using Data_mining_project.Extensions;
#endregion DataminingNameSpaces

#region SharpLearningNameSpaces
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
#endregion SharpLearningNameSpaces

namespace Data_mining_project.PostPruners
{
    public sealed class ErrorComplexityPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {

            throw new NotImplementedException();
        }

        /// <summary>
        /// Calculates the error complexity of <paramref name="node"/>.
        /// </summary>
        /// <param name="t">The binary tree in which the <paramref name="node"/> is contained.</param>
        /// <param name="trainSet">The <see cref="ObservationTargetSet"/> used to calculate error rates.</param>
        /// <param name="node">The <see cref="Node"/> to calculate the error complexity of.</param>
        /// <returns>
        /// A <see cref="double"/>, indicating the potential gain of pruning this <paramref name="node"/>. 
        /// A lower result indicates a better <see cref="Node"/> to prune.
        /// </returns>
        private static double ErrorComplexity(BinaryTree t, ObservationTargetSet trainSet, Node node)
        {
            double nodeCost = NodeErrorCost(t, trainSet, node);
            double treeCost = TreeErrorCost(t, trainSet, node);
            int leafCount = t.GetLeaves(node).Count;

            return (nodeCost - treeCost) / leafCount;
        }

        /// <summary>
        /// Calculates the error cost of the subtree rooted at <paramref name="subRoot"/>.
        /// </summary>
        /// <param name="t">The <see cref="BinaryTree"/> in which the <paramref name="subRoot"/> is contained.</param>
        /// <param name="trainSet">The <see cref="ObservationTargetSet"/> used to calculate the misclassification rates.</param>
        /// <param name="subRoot">The node from which to start calculating the error cost.</param>
        /// <returns>A <see cref="double"/>, representing the error cost of the subtree rooted at <paramref name="subRoot"/>.</returns>
        private static double TreeErrorCost(BinaryTree t, ObservationTargetSet trainSet, Node subRoot)
        {
            //TODO: does a non-leaf node count towards the tree error cost?
            //      if so, we should not initialize this to zero, but to NodeErrorCost(t, trainSet, subRoot).
            double sum = 0;

            // The subRoot has a left child, so we must add its node error cost.
            if(subRoot.LeftIndex != -1)
            {
                sum += NodeErrorCost(t, trainSet, t.Nodes[subRoot.LeftIndex]);
            }
            
            // The subRoot has a right child, so we must add its node error cost.
            if(subRoot.RightIndex != -1)
            {
                sum += NodeErrorCost(t, trainSet, t.Nodes[subRoot.RightIndex]);
            }

            return sum;
        }
        
        /// <summary>
        /// Calculates the error cost of a single <see cref="Node"/>.
        /// </summary>
        /// <param name="t">The tree in which the <paramref name="node"/> is located.</param>
        /// <param name="trainSet">The <see cref="ObservationTargetSet"/> that contains the training data.</param>
        /// <param name="node">The <see cref="Node"/> for which the error cost will be calculated.</param>
        /// <returns>A double, which indicates the error cost of this node.</returns>
        private static double NodeErrorCost(BinaryTree t, ObservationTargetSet trainSet, Node node)
        {
            //TODO: does a non-leaf node have a node error cost?
            //      if not, we should add an if statement that returns 0d when the node is not a leaf node.
            //      i *think* it does, but i am not entirely sure.
            int totalExamples = trainSet.Targets.Length;

            F64Matrix populations = t.Populations(trainSet);
            double[] nodeClasses = populations.Row(node.NodeIndex);

            double misclassificationcount = nodeClasses.Sum() - nodeClasses.Where(v => v.Equals(node.Value)).Count();
            
            return misclassificationcount / totalExamples;
        }
    }
}
