#include <limits.h>
#include <math.h>

#include "LKH.h"

/*
 * ParetoLab extension of the NeuroLKH SWIG wrapper.
 *
 * Upstream ReadProblem always builds a Euclidean cost matrix from the
 * coordinates stored at the front of invec. Those coordinates are only the
 * input of SparseGCNModel. When invec also carries a full integer matrix,
 * this function copies that matrix into CostMatrix so the Lin-Kernighan
 * search optimizes the explicit symmetric costs.
 *
 * Layout, with N = num_nodes:
 *   [0, 2N)                         coordinates (network / Euclidean fallback)
 *   [2N, 7N)                        5 candidate indices per node
 *   [7N, 8N)                        node penalties
 *   [8N, 8N + N*N)                  row-major integer costs
 *   anything after that             scratch space for per-trial statistics
 *
 * A zero block is ignored so the upstream Euclidean benchmark, which pads
 * invec with zeros, keeps its original distances.
 */

int ParetoLabExplicitCosts = 0;

void ParetoLabInstallExplicitCosts(double *invec, int length, int num_nodes)
{
    int base, need, i, j, k;
    int all_zero = 1;

    ParetoLabExplicitCosts = 0;
    if (invec == 0 || num_nodes < 2 || CostMatrix == 0 || length < 0)
        return;
    base = 8 * num_nodes;
    need = base + num_nodes * num_nodes;
    if (length < need)
        return;
    for (k = 0; k < num_nodes * num_nodes; k += 1) {
        if (invec[base + k] != 0.0) {
            all_zero = 0;
            break;
        }
    }
    if (all_zero)
        return;

    for (i = 2; i <= num_nodes; i += 1) {
        for (j = 1; j < i; j += 1) {
            long long rounded = llround(invec[base + (i - 1) * num_nodes + (j - 1)]);
            if (rounded > INT_MAX)
                rounded = INT_MAX;
            if (rounded < INT_MIN)
                rounded = INT_MIN;
            CostMatrix[(size_t) (i - 1) * (i - 2) / 2 + (j - 1)] = (int) rounded;
        }
    }
    ParetoLabExplicitCosts = 1;
}

void ParetoLabWriteTour(double *result, int length)
{
    int i;

    if (result == 0 || BestTour == 0 || DimensionSaved <= 0 || length < DimensionSaved)
        return;
    for (i = 0; i < DimensionSaved; i += 1)
        result[i] = (double) BestTour[i + 1];
    if (length > DimensionSaved)
        result[DimensionSaved] = (double) BestCost;
}
