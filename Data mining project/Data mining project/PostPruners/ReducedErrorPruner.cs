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
    public sealed class ReducedErrorPruner : IPruner
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
        public void Prune(Classifier c)
        {
            if (c.Model is not ClassificationDecisionTreeModel m)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a model, call {nameof(c.Learn)} first!");
            }

            if (c.TrainSet is not ObservationTargetSet trainSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a training data set, which is required for reduced error pruning!");
            }

            if (c.PruneSet is not ObservationTargetSet pruneSet)
            {
                throw new InvalidOperationException($"{nameof(c)} does not have a pruning data set, which is required for reduced error pruning!");
            }

            BinaryTree t = m.Tree;

            // Matrix that stores the population of the classes at each node.
            F64Matrix populations = this.Populations(trainSet, t);

            for (int i = t.Nodes.Count - 1; i >= 0; i--)
            {
                Node oldNode = t.Nodes[i];

                // Only proceed if this is a non-leaf node
                if (oldNode.LeftIndex != -1 && oldNode.RightIndex != -1)
                {
                    double prePrunedError = this.PruneSetError(m, pruneSet);

                    // Find the most frequent class of this node using the populations matrix
                    double[] nodeClasses = populations.Row(i);
                    double frequencyOfmostFrequentClass = nodeClasses.Max();
                    int indexMostFrequent = Array.IndexOf(nodeClasses, frequencyOfmostFrequentClass);
                    double mostFrequentClass = t.TargetNames[indexMostFrequent];

                    // Create a new node which is basically a copy of the old node but without childeren.
                    t.Nodes[i] = new Node(
                        -1, // Feature index is set to -1, this indicates that this is a leaf
                        mostFrequentClass, // Most populous class
                        -1, // The index of the left and right child shouldn't matter.
                        -1,
                        oldNode.NodeIndex,
                        oldNode.LeafProbabilityIndex
                    );


                    // Now if the accuracy has stayed the same or has improved, keep the change. Otherwise, we put back the old node.
                    if (this.PruneSetError(m, pruneSet) > prePrunedError)
                    {
                        t.Nodes[i] = oldNode; // Revert change
                    }
                }
            }
        }

        /// <summary>
        /// Calculates the populations of each node. <br/>
        /// The population of a node is defined as the number of observations that pass through a given node.
        /// We store this in a two-dimensional <see cref="F64Matrix"/>, 
        /// where the first index is the node index and the second index is the class index.
        /// </summary>
        /// <param name="t">The binary tree to calculate the populations for.</param>
        /// <param name="trainSet">The set of training data, that will be used to obtain the observations.</param>
        /// <returns>The <see cref="F64Matrix"/> that contains the populations of <paramref name="t"/>.</returns>
        private F64Matrix Populations(ObservationTargetSet trainSet, BinaryTree t)
        {
            int rows = t.Nodes.Count;
            // This is the amount of classes that exist in the data set
            // Each row in the matrix represents a node and each column is the amount of observations of this class that passed through this node.
            int cols = t.TargetNames.Length;

            F64Matrix populations = new(rows, cols);
            Node rootTrainNode = t.Nodes[0];
            for (int i = 0; i < trainSet.Targets.Length; i++)
            {
                // For every row in the training set, we descend the tree and increment the population of the node we end up at.
                double[] Xi = trainSet.Observations.Row(i);
                double yi = trainSet.Targets[i];
                this.AddXToPopulations(rootTrainNode, Xi, yi, populations, t);
            }

            return populations;
        }

        /// <summary>
        /// Increments the population matrix for the provided node. <br/>
        /// Then, descends further into the tree, depending on the feature of the node and the value of the observation. <br/>
        /// This method is based on <see cref="BinaryTree"/>, which can also be found at
        /// <seealso href=" https://github.com/mdabros/SharpLearning/blob/3f6063fad8886b09c5715c8713541045be7560b9/src/SharpLearning.DecisionTrees/Nodes/BinaryTree.cs"/>.
        /// </summary>
        /// <param name="node">The <see cref="Node"/> to start at.</param>
        /// <param name="observation">The observation values to use.</param>
        /// <param name="populations">The <see cref="F64Matrix"/> to store the results in.</param>
        /// <param name="t">The <see cref="BinaryTree"/> that <paramref name="node"/> belongs to.</param>
        /// <param name="target">The target value for the <paramref name="observation"/>.</param>
        private void AddXToPopulations(Node node, double[] observation, double target, F64Matrix populations, BinaryTree t)
        {
            // The node is a leaf node, so we don't need to do anything.
            if (node.FeatureIndex == -1.0)
            {
                return;
            }

            // This node is not an empty leaf, so we must update the populations matrix.
            // First, we determine the array index that should be incremented.
            int rowIndex = node.NodeIndex;

            int colIndex = Array.IndexOf(t.TargetNames, target);

            // Having determined this, we now add one to the count of this class to the populations matrix.
            populations[rowIndex, colIndex] = populations[rowIndex, colIndex] + 1;

            //If this is the case, the tree would choose the left node, so we need to descend into the left node.
            if (observation[node.FeatureIndex] <= node.Value)
            {
                this.AddXToPopulations(t.Nodes[node.LeftIndex], observation, target, populations, t);
                return;
            }
            //And otherwise, we have to descend into the right node.
            else
            {
                this.AddXToPopulations(t.Nodes[node.RightIndex], observation, target, populations, t);
                return;
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
