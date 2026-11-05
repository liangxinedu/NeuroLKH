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

    /* ParetoLab: the network input is always 20 slots. ReadProblem clamps
       MaxCandidates to Dimension - 1, so small instances must repeat neighbors. */
    {
	int slots = 20;
	for (int i = 1; i <= n_nodes; i += 1) {
	    NN = NodeSet[i].CandidateSet;
	    int have_last = 0;
	    double last_index = 0;
	    double last_dist = 0;
	    double last_inverse = -1;
	    for (int j = 0; j < slots; j += 1) {
		if (NN && NN->To) {
		    invec[(i - 1) * slots + j] = (double) (NN->To->Id - 1);
		    invec[n_nodes * slots + (i - 1) * slots + j] = D(&NodeSet[i], NN->To);
		    int Top = 0;
		    Node *NTo = &NodeSet[NN->To->Id];
		    Candidate *ToCandidateSet = NTo->CandidateSet;
		    for (int l = 0; l < slots; l += 1) {
			if (!ToCandidateSet || !ToCandidateSet->To)
			    break;
			if (ToCandidateSet->To->Id == i) {
			    invec[n_nodes * slots * 2 + (i - 1) * slots + j] = (double) ((NN->To->Id - 1) * slots + l);
			    Top = 1;
			    break;
			}
			ToCandidateSet += 1;
		    }
		    if (Top == 0)
			invec[n_nodes * slots * 2 + (i - 1) * slots + j] = (double) -1;
		    last_index = invec[(i - 1) * slots + j];
		    last_dist = invec[n_nodes * slots + (i - 1) * slots + j];
		    last_inverse = invec[n_nodes * slots * 2 + (i - 1) * slots + j];
		    have_last = 1;
		    NN += 1;
		} else if (have_last) {
		    invec[(i - 1) * slots + j] = last_index;
		    invec[n_nodes * slots + (i - 1) * slots + j] = last_dist;
		    invec[n_nodes * slots * 2 + (i - 1) * slots + j] = last_inverse;
		} else {
		    invec[(i - 1) * slots + j] = 0;
		    invec[n_nodes * slots + (i - 1) * slots + j] = 0;
		    invec[n_nodes * slots * 2 + (i - 1) * slots + j] = -1;
		}
	    }
	}
    }
    return (int) ((GetTime() - StartTime) * 1000000);
}
