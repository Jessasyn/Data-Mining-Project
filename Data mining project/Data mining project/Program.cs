using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers;
using SharpLearning.Metrics.Regression;
using Data_mining_project;

namespace Data_Mining_Project
{
    public sealed class Project
    {
        public static void Main(string[] args)
        {
            Classifier c = new Classifier(() => new StreamReader("winequality-white.csv"), "quality");
            c.ReadData(0.7);
            c.Learn();
            c.Predict();

            Console.WriteLine($"Test error: {c.TestError}");
            foreach (KeyValuePair<string, double> kvp in c.VariableImportance)
            {
                Console.WriteLine($"{kvp.Key},{kvp.Value}");
            }
        }
    }
}