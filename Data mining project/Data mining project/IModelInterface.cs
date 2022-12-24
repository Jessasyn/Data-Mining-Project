
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Models;

namespace Data_mining_project
{
    /// <summary>
    /// The interface that represents a classifier.
    /// </summary>
    public interface IModelInterface
    {
        /// <summary>
        /// Reads in data from the path provided during class construction, splits it with the specified <paramref name="trainPercentage"/>, 
        /// and stores the resulting sets in <see cref="TrainSet"/> and <see cref="TestSet"/>.
        /// </summary>
        /// <param name="trainPercentage">The percentage of data that should be put in the training set.</param>
        public void ReadData(double trainPercentage, double prunePercentage = 0d);

        /// <summary>
        /// Learns the <see cref="Model"/> of this classifier.
        /// Changing any parameter fields will not have an effect after this function is called! <br/>
        /// Also note that <see cref="ReadData(double)"/> <b>has</b> to be called before this function.
        /// </summary>
        /// <exception cref="InvalidOperationException">If <see cref="TrainSet"/> is not initialized.</exception>
        public void Learn();

        /// <summary>
        /// Predicts the test set, and reports the error measures. <br/>
        /// This function does <b>not</b> work if <see cref="Learn"/> has not been called.
        /// </summary>
        public void Error();

        /// <summary>
        /// Getter for the <see cref="ClassificationDecisionTreeModel"/> of this classifier.
        /// </summary>
        /// 
        public ClassificationDecisionTreeModel? GetModel();

        /// <summary>
        /// Getter for the <see cref="ObservationTargetSet"/> that is used for pruning.
        /// </summary>
        public ObservationTargetSet? GetPruneSet();

        /// <summary>
        /// Getter for the <see cref="ObservationTargetSet"/> that is used for training.
        /// </summary>
        public ObservationTargetSet? GetTrainingSet();

        /// <summary>
        /// Getter for the <see cref="ObservationTargetSet"/> that is used for testing.
        /// </summary>
        public ObservationTargetSet? GetTestSet();
    }
}
