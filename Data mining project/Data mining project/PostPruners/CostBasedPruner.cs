using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.PostPruners
{
    internal class CostBasedPruner : PrunerBase
    {
        public Dictionary<double, (double, double)> Costs { get; set; }
        public override void Prune(IClassifier c)
        {
            var rep = new ReducedErrorPruner();
        }
    }
}
