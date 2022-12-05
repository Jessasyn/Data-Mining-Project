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

            // TODO: This matrix is supposed to hold how many objects with a target pass through a certain node
            // However, we don't really know how many classes exist, so it is better to replace this with an array of dictionaries probably.
            // I've commented out a bunch of code for now to make it compilable but don't delete it.
/*            F64Matrix populations = Populations();

            void AddToPopulations(Node node, double[] observation, int target, F64Matrix populations)
            {
                // Based on method predict in https://github.com/mdabros/SharpLearning/blob/3f6063fad8886b09c5715c8713541045be7560b9/src/SharpLearning.DecisionTrees/Nodes/BinaryTree.cs#L11
                if (node.FeatureIndex == -1.0)
                {
                    return;
                }

                // Add this to the populations
                populations.

                if (observation[node.FeatureIndex] <= node.Value)
                {
                    
                }
                else
                {
                    return AddToPopulations(t.Nodes[node.RightIndex], observation, populations);
                }

                throw new InvalidOperationException("The tree is degenerated.");
            }
            /// <summary>
            /// For all nodes in the pruneSet,
            /// populations[node nr] = amount of observations of each class that pass through this node
            /// So this includes only non-leaves.
            /// </summary>
            /// <param name="t"></param>
            /// <param name="pruneSet"></param>
            /// <returns></returns>
            F64Matrix Populations()
            {
                // Based on: (But heavily modified)
                var rows = t.Nodes.Count;
                var cols = t.TargetNames.Length; // This is the amount of classes that exist in the data set
                // Each row in the matrix represents a node and each column is the amount of observations of this class that passed through this node.
                F64Matrix populations = new(rows, cols);
                for (int i = 0; i < c.TrainSet.Targets.Length; i++)
                    AddToPopulations(t.Nodes[0], c.TrainSet.Observations.Row(i), c.TrainSet.Targets[i] populations);
                return populations;
            }*/
        };
    }
}
