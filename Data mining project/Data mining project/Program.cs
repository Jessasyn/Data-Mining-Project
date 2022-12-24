#region DataMiningNameSpaces
using Data_mining_project.PostPruners;
using Data_mining_project.ModelInterfaces;
using Data_mining_project.Metrics;
using SharpLearning.Metrics.Classification;
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
            Console.WriteLine("ORDINAL WINE PROBLEM");
            OrdinalWine();

            Console.WriteLine("NOMINAL DIABETES PROBLEM");
            NominalDiabetes();
        }

        public static void NominalDiabetes()
        {
            // For a healthy person we really mind a false positive but a false negative not as much 
            // For a sick person we don't really mind a false positive but we really mind a false negative
            // The opposite is true for a sick person
            Dictionary<double, (double, double)> costs = new()
            { {0d, (1d, 0.2d)},
              {1d, (0.2d, 1d)} };

            //We run on all classifiers, starting with no pruning.
            Console.WriteLine("No pruning:");
            NominalClassificationModel noPrune = new NominalClassificationModel(DiabetesPath, DiabetesColumn);
            noPrune.ReadData(0.7);
            noPrune.Learn();
            ErrorAndCostError(noPrune, costs);


            //Reduced Error pruning
            Console.WriteLine("Reduced error pruning:");
            NominalClassificationModel redEr = new NominalClassificationModel(DiabetesPath, DiabetesColumn, new ReducedErrorPruner());
            redEr.ReadData(0.6, 0.2);
            redEr.Learn();
            ErrorAndCostError(redEr, costs);

            //Minimum Error pruning
            Console.WriteLine("Minimum error pruning:");
            NominalClassificationModel minEr = new NominalClassificationModel(DiabetesPath, DiabetesColumn, new MinimumErrorPruner());
            minEr.ReadData(0.6);
            minEr.Learn();
            ErrorAndCostError(minEr, costs);

            //Error complexity pruning
            Console.WriteLine("Error complexity pruning:");
            NominalClassificationModel erComp = new NominalClassificationModel(DiabetesPath, DiabetesColumn, new ErrorComplexityPruner());
            erComp.ReadData(0.6);
            erComp.Learn();
            ErrorAndCostError(erComp, costs);

            //Cost Based Pruning
            Console.WriteLine("Cost based pruning:");
            NominalClassificationModel costBased = new NominalClassificationModel(DiabetesPath, DiabetesColumn, new CostBasedPruner(costs));
            costBased.ReadData(0.6, 0.2);
            costBased.Learn();
            ErrorAndCostError(costBased, costs);
        }

        public static void ErrorAndCostError(NominalClassificationModel model, Dictionary<double, (double, double)> costs)
        {
            model.Error();
            model.CostError(costs);
            Console.WriteLine($"\tTotal classification Error: {model.TestError}");
            Console.WriteLine($"\tCost based error: {model.CostTestError}");
            Console.WriteLine($"\tTime taken: {model.PruneTime}");
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

            // Reduced error pruning
            Console.WriteLine("Reduced error pruning:");
            OrdinalClassificationModel reducedErrorPrune = new OrdinalClassificationModel(WinePath, WineTargetColumn, new ReducedErrorPruner());
            reducedErrorPrune.ReadData(0.3, 0.3);
            reducedErrorPrune.Learn();
            reducedErrorPrune.Error();
            Console.WriteLine($"\tError: {reducedErrorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {reducedErrorPrune.PruneTime}");

            // Minimum error pruning
            Console.WriteLine("Minimum error pruning:");
            OrdinalClassificationModel minimumErrorPrune = new OrdinalClassificationModel(WinePath, WineTargetColumn, new MinimumErrorPruner());
            minimumErrorPrune.ReadData(0.3, 0.3);
            minimumErrorPrune.Learn();
            minimumErrorPrune.Error();
            Console.WriteLine($"\tError: {minimumErrorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {minimumErrorPrune.PruneTime}");

            //Error complexity pruning
            Console.WriteLine("Error complexity pruning:");
            NominalClassificationModel errorPrune = new NominalClassificationModel(WinePath, WineTargetColumn, new ErrorComplexityPruner());
            errorPrune.ReadData(0.6);
            errorPrune.Learn();
            errorPrune.Error();
            Console.WriteLine($"\tError: {errorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {errorPrune.PruneTime}");
        }
    }
}