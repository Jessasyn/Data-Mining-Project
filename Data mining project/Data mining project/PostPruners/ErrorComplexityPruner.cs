using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    public sealed class ErrorComplexityPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {
            throw new NotImplementedException();
        }

        public double CalculateErrorCost(double[] observations, Node node, ObservationTargetSet testSet)
        {
            //R(t) = \frac{no of examples misclassified in node t \cdot no of examples in node t}{no of examples in node t \cdot no of totalexamples}
            //R(t) = \frac{no of examples misclassified in node t}{no of total examples}
            int totalExamples = testSet.Targets.Length;

            //TODO: take populations; 
            return 0d;
        }
    }
}
