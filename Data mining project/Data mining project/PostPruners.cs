using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project
{
    static public class PostPruners
    {
        static public Action<Classifier> ReducedError = (Classifier c) => 
        {
            ClassificationDecisionTreeModel m = c.Model ?? throw new NullReferenceException("Model not present.");
            // TODO: consider not using a null reference exception, and making the message more explicit.
            // something like 'invalidstatexception' is more proper (in that the state of the classifier does not allow for pruning,
            // whereas a null reference exception actually implies that some null pointer was dereferenced, which is not happening here.
            // then as to the message; the model is not present, but you can also do something like $"Classifier does not have a model, call {c.Learn} first!"
            // this then immediately tells any potential other developers (i.e., me :-) ), what is going wrong and how to fix it.

            if (c.PruneSet == null) { throw new NullReferenceException("Null pruning dataset for Reduced error Pruning"); }
            // TODO: same here with the null reference exception and its message.
            // TODO: youll want to use 'if null'. this guarantees no user defined operators are invoked, because those could yield unwanted null reference exceptions.
            // TODO: you can do one-line if statements, but then you probably want to omit the curly brackets; they imply a new scope, which you aren't using.

            BinaryTree t = m.Tree;

            List<int> populations = new List<int>(); // How many data points go through each node.
        };
    }
}
