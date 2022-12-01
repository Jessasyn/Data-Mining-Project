#region SharpLearningNameSpaces
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers;
#endregion SharpLearningNameSpaces

#region GenericNameSpaces
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
#endregion GenericNameSpaces

namespace Data_mining_project
{
    /// <summary>
    /// A wrapper around functionality from SharpLearning, to make it easier to use.
    /// </summary>
    public sealed class Classifier
    {
        /// <summary>
        /// The parser used for reading in datasets.
        /// </summary>
        private readonly CsvParser _parser;

        /// <summary>
        /// The name of the column that will be predicted to.
        /// </summary>
        private readonly string _targetColumn;

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
        public Dictionary<string, double>? VariableImportance { get; private set; }

        /// <summary>
        /// The test error of this classifier, for the specified dataset.
        /// </summary>
        public double? TestError { get; private set; }

        /// <summary>
        /// Action that handles post pruning when executed. <br/>
        /// The default value of this field is the empty lambda, which does not do anything, but it might be assigned some other pruner from elsewhere.
        /// </summary>
        public Action<Classifier> PostPruner = (Classifier classifier) => { };

        /// <summary>
        /// Constructs a new classifier, which will read from the path provided in <paramref name="parserPath"/>, 
        /// and consider the target column with the name <paramref name="targetColumn"/>.
        /// </summary>
        /// <param name="parserPath">The function which returns a textreader, to be used in the reading of the CSV.</param>
        /// <param name="targetColumn">The name of the column that contains the values to be predicted.</param>
        public Classifier(Func<TextReader> parserPath, string targetColumn)
        {
            //TODO: i propose the creation of an enum that is passed along in the constructor, which specifies the pruning method to use.
            // then, we can internalize that, which leads to more abstraction.
            this._parser = new CsvParser(parserPath);
            this._targetColumn = targetColumn;
        }

        /// <summary>
        /// Reads in data from the path provided during class construction, splits it with the specified <paramref name="trainPercentage"/>, 
        /// and stores the resulting sets in <see cref="TrainSet"/> and <see cref="TestSet"/>.
        /// </summary>
        /// <param name="trainPercentage">The percentage of data that should be put in the training set.</param>
        [MemberNotNull(nameof(this.TrainSet), nameof(this.TestSet))]
        public void ReadData(double trainPercentage, double prunePercentage=0d)
        {
            //Also known as y.
            double[] targets = this._parser.EnumerateRows(this._targetColumn)
                                           .ToF64Vector();

            //Also known as X.
            F64Matrix observations = this._parser.EnumerateRows(c => c != this._targetColumn)
                                                 .ToF64Matrix();

            //TODO: here would be a good spot for a comment that explains *why* we distinguish cases on the prunepercentage's value.
            if (prunePercentage > 0d)
            {
                PruningSetSplitter<double> splitter = new PruningSetSplitter<double>(trainPercentage, prunePercentage);
                PruningSetSplit pruningSetSplit = splitter.SplitSet(observations, targets);
                this.TrainSet = pruningSetSplit.TrainingSet;
                this.TestSet = pruningSetSplit.TestSet;
                this.PruneSet = pruningSetSplit.PruningSet;
            }
            else
            {
                TrainingTestIndexSplitter<double> splitter = new StratifiedTrainingTestIndexSplitter<double>(trainPercentage);
                TrainingTestSetSplit pruningSetSplit = splitter.SplitSet(observations, targets);
                this.TrainSet = pruningSetSplit.TrainingSet;
                this.TestSet = pruningSetSplit.TestSet;
            }
        }

        /// <summary>
        /// Learns the <see cref="Model"/> of this classifier.
        /// Changing any parameter fields will not have an effect after this function is called! <br/>
        /// Also note that <see cref="ReadData(double)"/> <b>has</b> to be called before this function.
        /// </summary>
        /// <exception cref="InvalidOperationException">If <see cref="TrainSet"/> is not initialized.</exception>
        [MemberNotNull(nameof(this.Model))]
        public void Learn()
        {
            if (this.TrainSet is null)
            {
                throw new InvalidOperationException($"Cannot call {this.Learn} before {this.ReadData} has been called!");
            }

            ClassificationDecisionTreeLearner treeLearner = new ClassificationDecisionTreeLearner(this.MaximumTreeDepth,  
                                                                                                  this.MinimumSplitSize, 
                                                                                                  this.FeaturesPerSplit,  
                                                                                                  this.MinimumInformationGain, 
                                                                                                  this.RandomSeed);

            this.Model = treeLearner.Learn(this.TrainSet.Observations, this.TrainSet.Targets);

            this.PostPruner(this);
        }

        /// <summary>
        /// Predicts the test set, and reports the error measures. <br/>
        /// This function does <b>not</b> work if <see cref="Learn"/> has not been called.
        /// </summary>
        [MemberNotNull(nameof(this.TestError), nameof(this.VariableImportance))]
        public void Predict()
        {
            if(this.Model is null || this.TestSet is null)
            {
                throw new InvalidOperationException($"Cannot call {this.Predict} before {this.Learn} has been called!");
            }

            double[] testPredictions = this.Model.Predict(this.TestSet.Observations);

            MeanSquaredErrorRegressionMetric metric = new MeanSquaredErrorRegressionMetric();
            this.TestError = metric.Error(this.TestSet.Targets, testPredictions);

            this.VariableImportance = this.Model.GetVariableImportance(this._parser.EnumerateRows(c => c != this._targetColumn)
                                                                                   .First().ColumnNameToIndex);
        }
    }
}
