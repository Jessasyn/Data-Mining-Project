using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace Data_mining_project
{
    /// <summary>
    /// Splitter that divides the sets into three partitions: for training, pruning and testing.
    /// </summary>
    /// <typeparam name="T">The type that is split.</typeparam>
    public struct PruningSetSplitter
    {
        /// <summary>
        /// The splitter that splits the training set off from the rest of the data.
        /// </summary>
        private readonly TrainingTestIndexSplitter<double> trainingSplitter;

        /// <summary>
        /// The splitter that splits the test set from the validation set.
        /// </summary>
        private readonly TrainingTestIndexSplitter<double> validationSplitter;

        /// <summary>
        /// Creates two internal stratified splitters, with the specified <paramref name="trainPercentage"/> and <paramref name="prunePercentage"/>. <br/>
        /// Note that the validation percentage is defined from the two other percentages.
        /// </summary>
        /// <param name="trainPercentage">The percentage of data that should be used for training</param>
        /// <param name="prunePercentage">The percentage of data that should be used for pruning</param>
        public PruningSetSplitter(double trainPercentage, double prunePercentage, int seed=42) 
        {
            if (trainPercentage <= 0d || trainPercentage >= 1d) throw new ArgumentException("Train percentage must be between 0 and 1");
            if (prunePercentage <= 0d || prunePercentage >= 1d) throw new ArgumentException("Prune percentage must be between 0 and 1");
            if (prunePercentage + trainPercentage >= 1d) throw new ArgumentException("The sum of the prune percentage and train percentage msust be between 0 and 1");

            this.trainingSplitter = new StratifiedTrainingTestIndexSplitter<double>(trainPercentage, seed);
            this.validationSplitter = new StratifiedTrainingTestIndexSplitter<double>(prunePercentage / (1 - trainPercentage), seed);
        }

        /// <summary>
        /// Splits a set into three sets, where the first two are always non-empty, and the last might be empty, 
        /// depending on the prune percentage set in the constructor.
        /// </summary>
        /// <param name="observations">The observations that should be split.</param>
        /// <param name="targets">The targets that should be splt.</param>
        /// <returns>The split sets, contained in <see cref="PruningSetSplit"/>.</returns>
        public PruningSetSplit SplitSet(F64Matrix observations, double[] targets)
        {
            TrainingTestSetSplit trainSplit = this.trainingSplitter.SplitSet(observations, targets);
            TrainingTestSetSplit pruneSplit = this.validationSplitter.SplitSet(trainSplit.TestSet.Observations, trainSplit.TestSet.Targets);

            ObservationTargetSet trainingSet = trainSplit.TrainingSet;
            ObservationTargetSet pruningSet = pruneSplit.TrainingSet;
            ObservationTargetSet testSet = pruneSplit.TestSet;

            return new PruningSetSplit(trainingSet, pruningSet, testSet);
        }
    }
}
