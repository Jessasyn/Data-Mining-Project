#region SharpLearningNameSpaces
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

#region DataminingNameSpaces
using Data_mining_project.Extensions;
using Data_mining_project.ModelInterfaces;
#endregion DataminingNameSpaces

namespace Data_mining_project.PostPruners
{
    public sealed class MinimumErrorPruner : PrunerBase
    {
        public override void Prune(IModelInterface c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if (c.GetTrainingSet() is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training data set, which is required for the {nameof(MinimumErrorPruner)}!");
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
                    if (PrunedNiblettBrotkoError(t, oldNode, populations) <= UnprunedNiblettBrotkoError(t, oldNode, populations))
                    {
                        double mostFrequentClass = t.MostFrequentClass(i, populations);
                        t.PruneNode(i, mostFrequentClass);
                    }
                }
            }
        }

        /// <summary>
        /// Calculate the Niblett-Brotko Error on a node <paramref name="t"/> in tree <paramref name="tree"/> using populations matrix <paramref name="populations"/>, 
        /// if that node is pruned into a leaf with its most popular class
        /// </summary>
        /// <param name="tree">The binary tree</param>
        /// <param name="t">The node</param>
        /// <param name="populations">The populations matrix</param>
        /// <returns>Niblett-Brotko Error E(<paramref name="t"/>)</returns>
        private static double PrunedNiblettBrotkoError(BinaryTree tree, Node t, F64Matrix populations)
        {
            int k = tree.TargetNames.Length;
            double[] nodePopulation = populations.Row(t.FeatureIndex);
            double nt = nodePopulation.Sum();
            double ntc = nodePopulation.Max();

            return (nt - ntc + k - 1) / (nt + k);
        }

        /// <summary>
        /// Calculate the Niblett-Brotko Error on a node <paramref name="t"/> in tree <paramref name="tree"/> using populations matrix <paramref name="populations"/>, 
        /// if that node is NOT pruned into a leaf with its most popular class
        /// </summary>
        /// <param name="tree">The binary tree</param>
        /// <param name="t">The node</param>
        /// <param name="populations">The populations matrix</param>
        /// <returns></returns>
        private static double UnprunedNiblettBrotkoError(BinaryTree tree, Node t, F64Matrix populations)
        {
            int k = tree.TargetNames.Length;
            double[] nodePopulation = populations.Row(t.FeatureIndex);
            double nt = nodePopulation.Sum();

            List<Node> leafNodes = tree.GetLeaves(tree.Nodes[t.FeatureIndex]);

            // Get the prediction of every leaf and add up the predictions.
            double[] subTreePredictions = new double[k];
            foreach (Node leafNode in leafNodes)
            {
                double[] leafPredictions = populations.Row(leafNode.NodeIndex);
                for (int j = 0; j < leafPredictions.Length; j++)
                {
                    subTreePredictions[j] += leafPredictions[j];
                }
            }
            double ntc = subTreePredictions.Max();

            return (nt - ntc + k - 1) / (nt + k);
        }
    }
}
