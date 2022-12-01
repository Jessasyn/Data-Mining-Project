using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace Data_mining_project
{
    /// <summary>
    /// Splitter that divides the sets into three partitions: for training, pruning and testing respectively
    /// </summary>
    internal sealed class PruningSetSplitter<T>
    {
        /// <summary>
        /// Splits data into three partitions: training, pruning and testing
        /// </summary>
        /// <param name="trainPercentage"></param>
        /// <param name="prunePercentage"></param>
        /// <returns></returns>
        /// 

        double trainPercentage;

        double prunePercentage;

        public PruningSetSplitter(double trainPercentage, double prunePercentage) 
        {
            this.trainPercentage = trainPercentage;
            this.prunePercentage = prunePercentage; 
            
        }
        public PruningSetSplit SplitSet(F64Matrix observations, double[] targets)
        {
            // Use two regular splitters provided by SharpLearning.
            TrainingTestIndexSplitter<double> trainSplitter = new StratifiedTrainingTestIndexSplitter<double>(trainPercentage);
            TrainingTestIndexSplitter<double> pruneSplitter = new StratifiedTrainingTestIndexSplitter<double>(prunePercentage / (1 - trainPercentage));
            // If we use the raw prune percentage, we get the fraction of the data that remains afters splitting off training

            TrainingTestSetSplit trainSplit = trainSplitter.SplitSet(observations, targets);
            TrainingTestSetSplit pruneSplit = pruneSplitter.SplitSet(trainSplit.TestSet.Observations, trainSplit.TestSet.Targets);

            ObservationTargetSet trainingSet = trainSplit.TrainingSet;
            ObservationTargetSet pruningSet = pruneSplit.TrainingSet;
            ObservationTargetSet testSet = pruneSplit.TestSet;

            return new PruningSetSplit(trainingSet, pruningSet, testSet);
        }
    }
}
