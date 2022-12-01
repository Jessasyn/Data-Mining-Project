using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace Data_mining_project
{
    /// <summary>
    /// Splitter that divides the sets into three partitions: for training, pruning and testing.
    /// </summary>
    /// <typeparam name="T">The type that is split.</typeparam>
    //TODO: this class is internal. that means its not able to be referenced outside of this project. inherently, there is nothing wrong with that, but 
    // the pruningsetsplit class is public, and used in here. it would make more sense if both are internal, or both public.
    internal sealed class PruningSetSplitter<T> //TODO: this T is not used anywhere. Im assuming you wish to use it in the stratified splitter?
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
        /// <param name="trainPercentage">The percentage of data that should </param>
        /// <param name="prunePercentage"></param>
        public PruningSetSplitter(double trainPercentage, double prunePercentage, int seed=42) 
        {
            //We create the two splitters, where the split percentage of the second is defined as the ratio between the prune percentage.
            //TODO: does this gracefully handle the case where prune = 0d?
            // it would be nice if it could...
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
            //TODO: we could also use an implicit tuple (ObservationTargetSet a, ObservationTargetSet b, ObservationTargetSet c) 
            //      that is not inherently better or worse, but if you choose to use a container, make it a struct! 
            //      you're not defining any methods or using generics, so its better for the sake of efficiency.
        }
    }
}
