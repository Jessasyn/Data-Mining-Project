#region DataminingNameSpaces
using Data_mining_project.Extensions;
#endregion DataminingNameSpaces

#region SharpLearningNameSpaces
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Metrics.Regression;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

namespace Data_mining_project.PostPruners
{
    public sealed class ErrorComplexityPruner : PrunerBase
    {
        /// <summary>
        /// The metric that is used to measure the error of the tree in <see cref="Prune(IModelInterface)"/>.
        /// </summary>
        private readonly MeanSquaredErrorRegressionMetric _metric = new MeanSquaredErrorRegressionMetric();

        /// <summary>
        /// Prunes the decision tree according to the rules of error complexity pruning.
        /// </summary>
        /// <param name="c">The classifier to use pruning on.</param>
        /// <exception cref="InvalidOperationException">If the state does not permit pruning.</exception>
        public override void Prune(IModelInterface c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if(c.GetTrainingSet() is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training set, which is required for the {nameof(ErrorComplexityPruner)}!");
            }

            List<List<Node>> forest = new List<List<Node>> { };

            BinaryTree t = m.Tree;

            forest.Add(t.Nodes.ToList());
            F64Matrix populations = t.Populations(trainSet);

            int childCount = t.GetChildren().Count;

            // We prune nodes, by the one that minimizes the error complexity metric, until there are (nearly) no nodes left.
            while (childCount > 3)
            {
                Node pruneNode = t.Nodes.Select(node => (node, ErrorComplexity(t, trainSet, node)))
                                        .MinBy(n => n.Item2).node;

                double nodeClass = t.MostFrequentClass(pruneNode, populations);

                int nodeChildren = t.GetChildren(pruneNode).Count;
                
                t.PruneNode(pruneNode.NodeIndex, nodeClass);

                forest.Add(m.Tree.Nodes.ToList());

                childCount -= nodeChildren;
            }

            List<double> accuracies = new List<double>();
            
            // Then, we calculate the accuracy of each, and take the tree with the maximum error.
            foreach(List<Node> tree in forest)
            {
                t.Nodes.Clear();
                t.Nodes.AddRange(tree);
                double[] pred = m.Predict(trainSet.Observations);
                accuracies.Add(this._metric.Error(trainSet.Targets, pred));
            }

            List<Node> bestTree = forest[accuracies.IndexOf(accuracies.Max())];

            // Lastly, we set the tree of the model to be this most accurate tree.
            t.Nodes.Clear();
            t.Nodes.AddRange(bestTree);
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
