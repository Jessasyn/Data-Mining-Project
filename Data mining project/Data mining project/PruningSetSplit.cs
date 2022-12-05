using SharpLearning.Containers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project
{
    public struct PruningSetSplit
    {
        public ObservationTargetSet TrainingSet;
        public ObservationTargetSet PruningSet;
        public ObservationTargetSet TestSet;
        public PruningSetSplit(ObservationTargetSet trainingSet, ObservationTargetSet pruningSet, ObservationTargetSet testSet) 
        {
            this.TrainingSet = trainingSet;
            this.PruningSet = pruningSet;
            this.TestSet = testSet;
        }
    }
}
