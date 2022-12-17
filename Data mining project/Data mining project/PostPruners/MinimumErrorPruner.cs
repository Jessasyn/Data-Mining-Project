#region SharpLearningNameSpaces
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.DecisionTrees.Nodes;

namespace Data_mining_project.PostPruners
{
    public sealed class MinimumErrorPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {
            
        }

        private double NiblettBrotkoError(Node t)
        {
            F64Matrix populations = populations();
        }
    }
}
