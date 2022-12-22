#region SharpLearningNameSpaces
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

namespace Data_mining_project.Extensions
{
    /// <summary>
    /// The class that contains several extension methods to the <see cref="BinaryTree"/> class.
    /// </summary>
    public static class BinaryTreeExtensions
    {
        /// <summary>
        /// Calculates the populations of each node. <br/>
        /// The population of a node is defined as the number of observations that pass through a given node.
        /// We store this in a two-dimensional <see cref="F64Matrix"/>, 
        /// where the first index is the node index and the second index is the class index.
        /// </summary>
        /// <param name="t">The binary tree to calculate the populations for.</param>
        /// <param name="trainSet">The set of training data, that will be used to obtain the observations.</param>
        /// <returns>The <see cref="F64Matrix"/> that contains the populations of <paramref name="t"/>.</returns>
        public static F64Matrix Populations(this BinaryTree t, ObservationTargetSet trainSet)
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
                populations.AddXToPopulations(rootTrainNode, Xi, yi, t);
            }

            return populations;
        }

        /// <summary>
        /// Find the most frequent class of the node with node number <paramref name="i"/> using the <paramref name="populations"/> matrix.
        /// </summary>
        /// <param name="t">The binary tree to which the <paramref name="populations"/> matrix belongs.</param>
        /// <param name="i">The index of the <see cref="Node"/> to get the most frequent class for.</param>
        /// <param name="populations">The populations matrix associated to the binary tree<paramref name="t"/>.</param>
        /// <returns>A <see cref="double"/>, representing the most frequent class at the node with index <paramref name="i"/>.</returns>
        public static double MostFrequentClass(this BinaryTree t, int i, F64Matrix populations)
        {
            double[] nodeClasses = populations.Row(i);
            double frequencyOfmostFrequentClass = nodeClasses.Max();
            int indexMostFrequent = Array.IndexOf(nodeClasses, frequencyOfmostFrequentClass);
            return t.TargetNames[indexMostFrequent];
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
        private static void AddXToPopulations(this F64Matrix populations, Node node, double[] observation, double target, BinaryTree t)
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
                populations.AddXToPopulations(t.Nodes[node.LeftIndex], observation, target, t);
                return;
            }
            //And otherwise, we have to descend into the right node.
            else
            {
                populations.AddXToPopulations(t.Nodes[node.RightIndex], observation, target, t);
                return;
            }
        }

        /// <summary>
        /// Prunes a single node from the <see cref="BinaryTree"/> <paramref name="t"/>.
        /// </summary>
        /// <param name="index">The index of the node to prune.</param>
        /// <param name="class">The leaf to change this node to.</param>
        /// <param name="t">The tree to prune from.</param>
        public static void PruneNode(this BinaryTree t, int index, double @class)
        {
            Node oldNode = t.Nodes[index];

            t.Nodes[index] = new Node(
                        -1, // Feature index is set to -1, this indicates that this is a leaf
                        @class, // Most populous class
                        -1, // The index of the left and right child shouldn't matter.
                        -1,
                        oldNode.NodeIndex,
                        oldNode.LeafProbabilityIndex
                    );
        }
    }
}
