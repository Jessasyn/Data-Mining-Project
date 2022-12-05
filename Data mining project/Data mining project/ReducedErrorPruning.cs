using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project
{
    static public class PostPruners
    {
        static public Action<Classifier> ReducedError = (Classifier c) => 
        {
            ClassificationDecisionTreeModel m = c.Model ?? throw new InvalidOperationException($"Classifier does not have a model, call classifier.Learn first!.");

            if (c.PruneSet is null) throw new InvalidOperationException($"Classifier does not have a pruning data set which is required for reduced error pruning");

            BinaryTree t = m.Tree;
            ObservationTargetSet pruneSet = c.PruneSet;

            List<int> populations = Populations(t, pruneSet); // How many data points go through each node.

            // For each object, we have to determine through which nodes it goes
        };

        static List<int> Populations(BinaryTree t, ObservationTargetSet pruneSet)
        {
            int[] populations = new int[t.Nodes.Count];
            return populations.ToList();
        }
    }
}
