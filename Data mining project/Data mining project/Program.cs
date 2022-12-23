#region DataMiningNameSpaces
using Data_mining_project.PostPruners;
using Data_mining_project;
#endregion DataMiningNameSpaces

namespace Data_Mining_Project
{
    public sealed class Project
    {
        public const string DataPath = @"winequality-white";

        public const string Column = "quality";
        
        public static void Main()
        {
            //We run on all classifiers, starting with no pruning.
            Console.WriteLine("No pruning:");
            Classifier noPrune = new Classifier(DataPath, Column);
            noPrune.ReadData(0.6);
            noPrune.Learn();
            noPrune.Predict();
            Console.WriteLine($"\tError: {noPrune.TestError}");
            Console.WriteLine($"\tTime taken: {noPrune.PruneTime}");

            Console.WriteLine("Error complexity pruning:");
            Classifier errorPrune = new Classifier(DataPath, Column, new ErrorComplexityPruner());
            errorPrune.ReadData(0.6);
            errorPrune.Learn();
            errorPrune.Predict();
            Console.WriteLine($"\tError: {errorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {errorPrune.PruneTime}");

            Console.WriteLine("Reduced error pruning:");
            Classifier reducedErrorPrune = new Classifier(DataPath, Column, new ReducedErrorPruner());
            reducedErrorPrune.ReadData(0.3, 0.3);
            reducedErrorPrune.Learn();
            reducedErrorPrune.Predict();
            Console.WriteLine($"\tError: {reducedErrorPrune.TestError}");
            Console.WriteLine($"\tTime taken: {reducedErrorPrune.PruneTime}");

            Console.WriteLine("Minimum error pruning:");
            Classifier minimumErrorPrune = new Classifier(DataPath, Column, new MinimumErrorPruner());
            minimumErrorPrune.ReadData(0.3, 0.3);
            minimumErrorPrune.Learn();
            minimumErrorPrune.Predict();
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