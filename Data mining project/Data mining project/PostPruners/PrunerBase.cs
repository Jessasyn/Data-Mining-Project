using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    /// <summary>
    /// Abstract base for a class that prunes.
    /// </summary>
    public abstract class PrunerBase : IPruner
    {
        public abstract void Prune(IClassifier c);
    }
}
