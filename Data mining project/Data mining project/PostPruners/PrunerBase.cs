using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// Abstract base for a class that prunes.
    /// </summary>
    public abstract class PrunerBase : IPruner
    {
        public abstract void Prune(IClassifier c);

        public void PruneNode(int index, double @class, BinaryTree t)
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
