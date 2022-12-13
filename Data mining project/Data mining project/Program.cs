﻿#region DataMiningNameSpaces
using Data_mining_project;
using Data_mining_project.PostPruners;
#endregion DataMiningNameSpaces

namespace Data_Mining_Project
{
    public sealed class Project
    {
        public static void Main()
        {
            Classifier c = new Classifier(() => new StreamReader("winequality-white.csv"), "quality", new ReducedErrorPruner());
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