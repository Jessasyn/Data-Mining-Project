#region SharpLearningNameSpaces
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.DecisionTrees.Nodes;
using Data_mining_project.Extensions;

namespace Data_mining_project.PostPruners
{
    public sealed class MinimumErrorPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {
            if (c.GetModel() is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if (c.GetTrainingSet() is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training data set, which is required for minimum error pruning!");
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
                        this.PruneNode(i, mostFrequentClass, t);
                    }
                }
            }
        }

        /// <summary>
        /// Calculate the Niblett-Brotko Error on a node <paramref name="t"/> using populations matrix <paramref name="populations"/>, 
        /// if that node is pruned into a leaf with its most popular class
        /// </summary>
        /// <param name="k">Total number of classes present in the population</param>
        /// <param name="t"></param>
        /// <param name="populations"></param>
        /// <returns>Niblett-Brotko Error E(<paramref name="t"/>)</returns>
        private double PrunedNiblettBrotkoError(BinaryTree tree, Node t, F64Matrix populations)
        {
            int k = tree.TargetNames.Length;
            double[] nodePopulation = populations.Row(t.FeatureIndex);
            double nt = nodePopulation.Sum();
            double ntc = nodePopulation.Max();

            return (nt - ntc + k - 1) / (nt + k);
        }

        private double UnprunedNiblettBrotkoError(BinaryTree tree, Node t, F64Matrix populations)
        {
            int k = tree.TargetNames.Length;
            double[] nodePopulation = populations.Row(t.FeatureIndex);
            double nt = nodePopulation.Sum();

            List<Node> leafNodes = new();
            getLeaves(tree, tree.Nodes[t.FeatureIndex], leafNodes);

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
        /// <summary>
        /// store all leaves of <paramref name="tree"/> in list <paramref name="leafNodes"/>.
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="node"></param>
        /// <param name="leafNodes"></param>
        private void getLeaves(BinaryTree tree, Node node, List<Node> leafNodes)
        {
            // The node is a leaf node, so we don't need to do anything.
            if (node.FeatureIndex == -1.0)
            {
                leafNodes.Add(node);
            }
            else
            {
                getLeaves(tree, tree.Nodes[node.RightIndex], leafNodes);
                getLeaves(tree, tree.Nodes[node.LeftIndex], leafNodes);
            }
        }
    }
}
