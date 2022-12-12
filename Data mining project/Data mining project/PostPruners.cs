using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project
{
    static public class PostPruners
    {
        static public Action<Classifier> ReducedError = (Classifier c) =>
        {
            ClassificationDecisionTreeModel m = c.Model ?? throw new InvalidOperationException($"Classifier does not have a model, call classifier.Learn first!.");

            if (c.TrainSet is null) throw new InvalidOperationException($"Classifier does not have a training data set which is required for reduced error pruning");

            if (c.PruneSet is null) throw new InvalidOperationException($"Classifier does not have a pruning data set which is required for reduced error pruning");

            BinaryTree t = m.Tree;
            ObservationTargetSet pruneSet = c.PruneSet;

            F64Matrix populations = Populations();

            /// <summary>
            /// Adds this observation to the populations matrix. See the documentation of Populations() for more info.
            /// </summary>
            /// <param name="node"></param>
            /// <param name="observation"></param>
            /// <param name="populations"></param>
            /// <returns></returns>
            void AddXToPopulations(Node node, double[] observation, double target, F64Matrix populations)
            {
                // This method is based on SharpLearning.DecisionTrees.Nodes.BinaryTree
                // https://github.com/mdabros/SharpLearning/blob/3f6063fad8886b09c5715c8713541045be7560b9/src/SharpLearning.DecisionTrees/Nodes/BinaryTree.cs
                if (node.FeatureIndex == -1.0) { return; } // Empty leaf case

                // This node is not an empty leaf, so add it to the populations!
                // Determine the array index that should be incremented
                int rowIndex = node.NodeIndex;

                double nodeClass = node.Value;
                int colIndex = Array.IndexOf(t.TargetNames, target);

                // Now add one to the count of this class to the populations matrix.
                populations[rowIndex, colIndex] = populations[rowIndex, colIndex] + 1;

                if (observation[node.FeatureIndex] <= node.Value) // Left child case
                {
                    AddXToPopulations(t.Nodes[node.LeftIndex], observation, target, populations);
                    return;
                }
                else // Right child case
                {
                    AddXToPopulations(t.Nodes[node.RightIndex], observation, target, populations);
                    return;
                }

                throw new InvalidOperationException("The tree is degenerated.");
            }

            /// <summary>
            /// Go through all the training data and store the population of each class per node in F64Matrix populations.
            /// 
            /// For example, if i is the index of the node n in t.Nodes,
            /// And j is the index of the class c in t.targetNames:
            /// Then populations[i, j] is the amount of observations of class c that passed through node n.
            /// 
            /// </summary>
            /// <returns>populations</returns>
            F64Matrix Populations()
            {
                var rows = t.Nodes.Count;
                var cols = t.TargetNames.Length; // This is the amount of classes that exist in the data set
                // Each row in the matrix represents a node and each column is the amount of observations of this class that passed through this node.
                F64Matrix populations = new(rows, cols);
                Node rootTrainNode = t.Nodes[0];
                for (int i = 0; i < c.TrainSet.Targets.Length; i++)
                {
                    // Using PredictNode, get the leaf node of each observation in the training set.
                    double[] Xi = c.TrainSet.Observations.Row(i);
                    double yi = c.TrainSet.Targets[i];
                    AddXToPopulations(rootTrainNode, Xi, yi, populations);
                }

                return populations;
            }
        };
    }
}
