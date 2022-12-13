using SharpLearning.Containers;

namespace Data_mining_project.Splitters
{
    /// <summary>
    /// Container class that stores the result of a split from <see cref="PruningSetSplit.Split(ObservationTargetSet, double)"/>.
    /// </summary>
    public struct PruningSetSplit
    {
        public ObservationTargetSet TrainingSet;
        public ObservationTargetSet PruningSet;
        public ObservationTargetSet TestSet;

        /// <summary>
        /// Constructs a new insance of <see cref="PruningSetSplit"/> from its component parts.
        /// </summary>
        /// <param name="trainingSet">The <see cref="ObservationTargetSet"/> that represents the training set.</param>
        /// <param name="pruningSet">The <see cref="ObservationTargetSet"/> that represents the pruning (or validation) set.</param>
        /// <param name="testSet">The <see cref="ObservationTargetSet"/> that represents the testing set.</param>
        public PruningSetSplit(ObservationTargetSet trainingSet, ObservationTargetSet pruningSet, ObservationTargetSet testSet)
        {
            TrainingSet = trainingSet;
            PruningSet = pruningSet;
            TestSet = testSet;
        }
    }
}
