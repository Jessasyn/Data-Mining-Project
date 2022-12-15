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
        public void Prune(Classifier c);
    }
}
