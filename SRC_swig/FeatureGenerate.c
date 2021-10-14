#include "LKH.h"
#include "Genetic.h"
#include "BIT.h"

/*
 * This file generates the feature for GCN
 * input: np array with size 100 x 2
 * output: edge_index 100 x 10
 *         edge_alpha 100 x 10
 *         inverse_edge_index 100 x 10
 *         node_order 100
 *         node_pi 100
 * inplace output
 */

int featureGenerate(int r_seed, double* invec, int n)
{
    ReadParameters();
    Seed = (unsigned) r_seed;
    MaxCandidates = 20;

    n_nodes = n / 60;
    ReadProblem(invec);
    GeneratingFeatures = 1;
    NeuroLKH = 0;
    StartTime = GetTime();
    AllocateStructures();
    CandidateSetType = NN;
    StartTime = GetTime();
    CreateNearestNeighborCandidateSet(MaxCandidates);

    Candidate *NN;

    for (int i = 1; i <= n_nodes; i += 1) {	
	NN = NodeSet[i].CandidateSet;
	for (int j = 0; j < MaxCandidates; j += 1) {
	    assert (NN->To);
            invec[(i - 1) * MaxCandidates + j] = (double) (NN->To->Id - 1);
	    invec[n_nodes * MaxCandidates + (i - 1) * MaxCandidates + j] = D(&NodeSet[i], NN->To);
	    int Top = 0;
	    Node *NTo = &NodeSet[NN->To->Id];
	    Candidate *ToCandidateSet = NTo->CandidateSet;
	    for (int l = 0; l < MaxCandidates; l += 1) {
                if (ToCandidateSet->To->Id == i) {
		    invec[n_nodes * MaxCandidates * 2 + (i - 1) * MaxCandidates + j] = (double) ((NN->To->Id - 1) * MaxCandidates + l);
		    Top = 1;
		    break;
                }
		ToCandidateSet += 1;
            }
	    if (Top == 0) {
	        invec[n_nodes * MaxCandidates * 2 + (i - 1) * MaxCandidates + j] = (double) -1;
	    }
	    NN += 1;
        }
    }
    return (int) ((GetTime() - StartTime) * 1000000);
}
