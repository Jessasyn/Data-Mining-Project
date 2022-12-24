#region SharpLearningNameSpaces
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers;
using SharpLearning.Metrics.Classification;
#endregion SharpLearningNameSpaces

#region GenericNameSpaces
using System.Diagnostics.CodeAnalysis;
#endregion GenericNameSpaces

#region DataminingNameSpaces
using Data_mining_project.Splitters;
using Data_mining_project.PostPruners;
#endregion DataminingNameSpaces

namespace Data_mining_project.ModelInterfaces
{
    /// <summary>
    /// A wrapper around functionality from SharpLearning, to make it easier to use.
    /// </summary>
    public abstract class ClassificationModelBase : IClassificationModel
    {
        /// <summary>
        /// The parser used for reading in datasets.
        /// </summary>
        protected readonly CsvParser parser;

        /// <summary>
        /// The name of the column that will be predicted to.
        /// </summary>
        protected readonly string targetColumn;

        /// <summary>
        /// The maximum tree depth that a tree is allowed to grow to.
        /// </summary>
        public int MaximumTreeDepth = 2000;

        /// <summary>
        /// The minimum amount of nodes that have to be present in a split.
        /// </summary>
        public int MinimumSplitSize = 1;

        /// <summary>
        /// The minimum amount of features that must be present in a split.
        /// </summary>
        public int FeaturesPerSplit = 0;

        /// <summary>
        /// The minimum amount of information gain that must bbe present per split.
        /// </summary>
        public double MinimumInformationGain = 1E-06;

        /// <summary>
        /// The seed that will be passed to randomize splitting.
        /// </summary>
        public int RandomSeed = 42;

        /// <summary>
        /// The training set for this classifier. This field is <see langword="null"/> until <see cref="ReadData(double)"/> is called.
        /// </summary>
        public ObservationTargetSet? TrainSet { get; private set; }

        /// <summary>
        /// The test set for this classifier. This field is <see langword="null"/> until <see cref="ReadData(double)"/> is called.
        /// </summary>
        public ObservationTargetSet? TestSet { get; private set; }

        /// <summary>
        /// The prune set for this classifier. Only set and used with reduced error pruning. This field is <see langword="null"/> until <see cref="ReadData(double)"/> is called.
        /// </summary>
        public ObservationTargetSet? PruneSet { get; private set; }

        /// <summary>
        /// The model that has learned the <see cref="TrainSet"/>. This field is <see langword="null"/> until <see cref="Learn"/> is called.
        /// </summary>
        public ClassificationDecisionTreeModel? Model { get; private set; }

        /// <summary>
        /// The variable importance for each feature.
        /// </summary>
        public Dictionary<string, double>? VariableImportance { get; set; }

        /// <summary>
        /// The test error of this classifier, for the specified dataset.
        /// </summary>
        public double? TestError { get; set; }

        /// <summary>
        /// The <see cref="IPruner"/> that handles post pruning when executed. <br/>
        /// The default value of this field is null, which means that no post pruning will be done.
        /// </summary>
        protected readonly IPruner? PostPruner;

        /// <summary>
        /// The time it has taken to prune the classifier.
        /// </summary>
        public TimeSpan PruneTime { get; private set; } = TimeSpan.Zero;

        /// <summary>
        /// Constructs a new classifier, which will read from the path provided in <paramref name="parserPath"/>, 
        /// and consider the target column with the name <paramref name="targetColumn"/>.
        /// </summary>
        /// <param name="parserPath">The path to the.</param>
        /// <param name="targetColumn">The name of the column that contains the values to be predicted.</param>
        /// <param name="postPruner">The <see cref="IPruner"/> that will be used to prune the classifier, after it has been learned.</param>
        public ClassificationModelBase(string parserPath, string targetColumn, IPruner? postPruner = null)
        {
            if (!parserPath.EndsWith(".csv"))
            {
                parserPath += ".csv";
            }

            parser = new CsvParser(() => new StreamReader($"Datasets/{parserPath}"));
            this.targetColumn = targetColumn;
            PostPruner = postPruner;
        }

        [MemberNotNull(nameof(TrainSet), nameof(TestSet))]
        public void ReadData(double trainPercentage, double prunePercentage = 0d)
        {
            //Also known as y.
            double[] targets = parser.EnumerateRows(targetColumn)
                                           .ToF64Vector();

            //Also known as X.
            F64Matrix observations = parser.EnumerateRows(c => c != targetColumn)
                                                 .ToF64Matrix();

            // If the optional parameter prunePercentage is non-zero, we want to reserve data for pruning.
            // To do so, we make a pruningset split, and then make a train/test split on the remaining data.
            if (prunePercentage > 0d)
            {
                PruningSetSplitter splitter = new PruningSetSplitter(trainPercentage, prunePercentage);
                PruningSetSplit pruningSetSplit = splitter.SplitSet(observations, targets);
                TrainSet = pruningSetSplit.TrainingSet;
                TestSet = pruningSetSplit.TestSet;
                PruneSet = pruningSetSplit.PruningSet;
            }
            // If the optional parameter prunePercentage is zero, we don't need to reserve data for pruning.
            // So, we can just make a train/test split on the entire dataset.
            else
            {
                TrainingTestIndexSplitter<double> splitter = new StratifiedTrainingTestIndexSplitter<double>(trainPercentage);
                TrainingTestSetSplit pruningSetSplit = splitter.SplitSet(observations, targets);
                TrainSet = pruningSetSplit.TrainingSet;
                TestSet = pruningSetSplit.TestSet;
            }
        }

        [MemberNotNull(nameof(Model))]
        public void Learn()
        {
            if (TrainSet is null)
            {
                throw new InvalidOperationException($"Cannot call {Learn} before {ReadData} has been called!");
            }

            ClassificationDecisionTreeLearner treeLearner = new ClassificationDecisionTreeLearner(MaximumTreeDepth,
                                                                                                  MinimumSplitSize,
                                                                                                  FeaturesPerSplit,
                                                                                                  MinimumInformationGain,
                                                                                                  RandomSeed);

            Model = treeLearner.Learn(TrainSet.Observations, TrainSet.Targets);

            DateTime start = DateTime.UtcNow;
            PostPruner?.Prune(this);
            DateTime end = DateTime.UtcNow;

            PruneTime = end - start;
        }

        [MemberNotNull(nameof(TestError), nameof(VariableImportance))]

        public abstract void Error();

        ClassificationDecisionTreeModel? IClassificationModel.GetModel() => Model;

        ObservationTargetSet? IClassificationModel.GetPruneSet() => PruneSet;

        ObservationTargetSet? IClassificationModel.GetTrainingSet() => TrainSet;

        ObservationTargetSet? IClassificationModel.GetTestSet() => TestSet;
    }
}
