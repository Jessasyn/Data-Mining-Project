#region DataMiningNameSpaces
using Data_mining_project.PostPruners;
using Data_mining_project.ModelInterfaces;
#endregion DataMiningNameSpaces

namespace Data_Mining_Project
{
    public sealed class Project
    {
        public const string WinePath = @"winequality-white";

        public const string WineTargetColumn = "quality";

        public const string DiabetesPath = @"diabetes";

        public const string DiabetesColumn = "Outcome";

        public static void Main()
        {
            NominalDiabetes();
        }

        public static void NominalDiabetes()
        {
            //We run on all classifiers, starting with no pruning.
            Console.WriteLine("No pruning:");
            NominalClassificationModel noPrune = new NominalClassificationModel(DiabetesPath, DiabetesColumn);
            noPrune.ReadData(0.6);
            noPrune.Learn();
            noPrune.Error();
            Console.WriteLine($"\tError: {noPrune.TestError}");
            Console.WriteLine($"\tTime taken: {noPrune.PruneTime}");
        }

        public static void OrdinalWine()
        {
            //We run on all classifiers, starting with no pruning.
            Console.WriteLine("No pruning:");
            OrdinalClassificationModel noPrune = new OrdinalClassificationModel(WinePath, WineTargetColumn);
            noPrune.ReadData(0.6);
            noPrune.Learn();
            noPrune.Error();
            Console.WriteLine($"\tError: {noPrune.TestError}");
            Console.WriteLine($"\tTime taken: {noPrune.PruneTime}");

            /*            Console.WriteLine("Error complexity pruning:");
                        RegressionModel errorPrune = new RegressionModel(WinePath, WineTargetColumn, new ErrorComplexityPruner());
                        errorPrune.ReadData(0.6);
                        errorPrune.Learn();
                        errorPrune.Error();
                        Console.WriteLine($"\tError: {errorPrune.TestError}");
                        Console.WriteLine($"\tTime taken: {errorPrune.PruneTime}");*/

            Console.WriteLine("Reduced error pruning:");
            OrdinalClassificationModel reducedErrorPrune = new OrdinalClassificationModel(WinePath, WineTargetColumn, new ReducedErrorPruner());
            reducedErrorPrune.ReadData(0.3, 0.3);
            reducedErrorPrune.Learn();
            reducedErrorPrune.Error();
            Console.WriteLine($"\tError: {reducedErrorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {reducedErrorPrune.PruneTime}");

            Console.WriteLine("Minimum error pruning:");
            OrdinalClassificationModel minimumErrorPrune = new OrdinalClassificationModel(WinePath, WineTargetColumn, new MinimumErrorPruner());
            minimumErrorPrune.ReadData(0.3, 0.3);
            minimumErrorPrune.Learn();
            minimumErrorPrune.Error();
            Console.WriteLine($"\tError: {minimumErrorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {minimumErrorPrune.PruneTime}");

            //TODO: figure out actual costs.
            //Console.WriteLine("Cost based pruning:");
            //Classifier costBasedPruner = new Classifier(DataPath, Column, new CostBasedPruner(new Dictionary<double, (double, double)> { }));
            //costBasedPruner.ReadData(0.3, 0.3);
            //costBasedPruner.Learn();
            //costBasedPruner.Predict();
            //Console.WriteLine($"\tError: {costBasedPruner.TestError}");
            //Console.WriteLine($"\tTime taken: {costBasedPruner.PruneTime}");
        }
    }
}