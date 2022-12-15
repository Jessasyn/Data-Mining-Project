using Data_mining_project.Extensions;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    public sealed class ErrorComplexityPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {
            throw new NotImplementedException();
        }

        public double NodeErrorCost(BinaryTree t, ObservationTargetSet trainSet, Node node)
        {
            int totalExamples = trainSet.Targets.Length;

            F64Matrix populations = t.Populations(trainSet);
            double[] nodeClasses = populations.Row(node.NodeIndex);

            double misclassificationcount = nodeClasses.Sum() - nodeClasses.Where(v => v.Equals(node.Value)).Count();
            
            return misclassificationcount / totalExamples;
        }
    }
}
