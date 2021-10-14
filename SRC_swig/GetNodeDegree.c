#include "LKH.h"
#include "Genetic.h"
#include "BIT.h"

/*
 * This file contains the main function of the program.
 * input : invec 100 x 2 coords,
 *               100 x 5 candidates, if deep
 *               100 for node orders, 100 for node Pi if read_pi
 */

void getNodeDegree(int r_seed,  double* invec, int n)
{
    GainType Cost, OldOptimum;
    double Time, LastTime = GetTime();
    Node *N;
    n_nodes = n / 3;

    ReadParameters();
    Subgradient = 0;
    GettingNodeDegree = 1;
    NeuroLKH = 0;
    Seed = (unsigned) r_seed;
    // MaxMatrixDimension = 20000;

    ReadProblem(invec);
    AllocateStructures();
    ReadPenalties(invec);
    CreateCandidateSet(invec);
}
