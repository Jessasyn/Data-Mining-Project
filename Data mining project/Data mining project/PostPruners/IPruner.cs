using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// The <see cref="IPruner"/> <see langword="interface"/> defines the functionality that a pruner class must implement.
    /// </summary>
    public interface IPruner
    {
        /// <summary>
        /// Prunes the <see cref="BinaryTree"/> present in <paramref name="c"/>.
        /// </summary>
        /// <param name="c">The classifier that will be pruned.</param>
        public void Prune(IClassifier c);

        /// <summary>
        /// Prunes a single node from the <see cref="BinaryTree"/> <paramref name="t"/>.
        /// </summary>
        /// <param name="index">The index of the node to prune.</param>
        /// <param name="class">The leaf to change this node to.</param>
        /// <param name="t">The tree to prune from.</param>
        public void PruneNode(int index, double @class, BinaryTree t);
    }
}
