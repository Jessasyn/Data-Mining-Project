using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_mining_project.PostPruners
{
    public sealed class MinimumErrorPruner : PrunerBase
    {
        public override void Prune(IClassifier c)
        {
            
        }

        private double NiblettBrotkoError()
        {
            return 0d;
        }
    }
}
