using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project
{
    static public class PostPruners
    {
        static public Action<Classifier> ReducedError = (Classifier c) => 
        {
            ClassificationDecisionTreeModel m = c.Model ?? throw new NullReferenceException("Model not present.");
            if (c.PruneSet == null) { throw new NullReferenceException("Null pruning dataset for Reduced error Pruning"); }
            BinaryTree t = m.Tree;

            List<int> populations = new List<int>(); // How many data points go through each node.
        };
    }
}
