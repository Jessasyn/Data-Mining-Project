#region DataMiningNameSpaces
using Data_mining_project.PostPruners;
using Data_mining_project;
#endregion DataMiningNameSpaces

namespace Data_Mining_Project
{
    public sealed class Project
    {
        public static void Main()
        {
            //Classifier c = new Classifier(() => new StreamReader("winequality-white.csv"), "quality", new ErrorComplexityPruner());
            //Classifier c = new Classifier(() => new StreamReader("winequality-white.csv"), "quality", new MinimumErrorPruner());
            //Dictionary<double, (double, double)> costs = new();
            Classifier c = new Classifier(() => new StreamReader("winequality-white.csv"), "quality", new CostBasedPruner(costs));
            //Classifier c = new Classifier(() => new StreamReader("winequality-white.csv"), "quality");
            c.ReadData(0.5, 0.3);
            c.Learn();
            c.Predict();

            Console.WriteLine($"Test error: {c.TestError}\n");

            foreach (KeyValuePair<string, double> kvp in c.VariableImportance)
            {
                Console.WriteLine($"{kvp.Key}: {kvp.Value}");
            }
        }
    }
}