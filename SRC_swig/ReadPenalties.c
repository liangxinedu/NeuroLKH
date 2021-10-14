#include "LKH.h"

/*
 * The ReadPenalties function attempts to read node penalties (Pi-values)
 * from file. 
 *
 * The first line of the file contains the number of nodes.
 *
 * Each of the following lines is of the form
 *       <integer> <integer>
 * where the first integer is a node number, and the second integer 
 * is the Pi-value associated with the node.
 *
 * If reading succeeds, the function returns 1; otherwise 0.
 *
 * The function is called from the CreateCandidateSet function. 
 */

int ReadPenalties(double* invec)
{
    int i, Id;
    Node *Na, *Nb = 0;
    static int PenaltiesRead = 0;
    if (NeuroLKH == 0 && GettingNodeDegree == 0)
	return 0;
//    if (PiFileName == 0)
//        return 0;
//    if (PenaltiesRead || !strcmp(PiFileName, "0"))
//        return PenaltiesRead = 1;
//    if (!(PiFile = fopen(PiFileName, "r")))
//        return 0;
//    if (TraceLevel >= 1)
//        printff("Reading PI_FILE: \"%s\" ... ", PiFileName);
//    fscanint(PiFile, &i);
//    if (i != Dimension)
//        eprintf("PI_FILE \"%s\" does not match problem", PiFileName);
//    fscanint(PiFile, &Id);
//    assert(Id >= 1 && Id <= Dimension);

    for (i = 1; i <= n_nodes; i++) {
        Nb = &NodeSet[i];
        if (NeuroLKH)
	    Nb->Pi = (int) invec[7 * n_nodes + i - 1];
	else if (GettingNodeDegree)
	    Nb->Pi = (int) invec[2 * n_nodes + i - 1];
    }

//    fclose(PiFile);
//    if (TraceLevel >= 1)
//        printff("done\n");
    return PenaltiesRead = 1;
}
