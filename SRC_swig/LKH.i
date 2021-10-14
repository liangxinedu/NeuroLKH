%module LKH


%{
#define SWIG_FILE_WITH_INIT
#include "INCLUDE/BIT.h"
#include "INCLUDE/Delaunay.h"
#include "INCLUDE/GainType.h"
#include "INCLUDE/Genetic.h"
#include "INCLUDE/GeoConversion.h"
#include "INCLUDE/Hashing.h"
#include "INCLUDE/Heap.h"
#include "INCLUDE/LKH.h"
#include "INCLUDE/Segment.h"
#include "INCLUDE/Sequence.h"
#include "INCLUDE/gpx.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* invec, int n)}

long long main(int deep, int num_runs, int r_seed, int num_nodes, double* invec, int n);
int featureGenerate(int r_seed, double* invec, int n);
void getNodeDegree(int r_seed, double* invec, int n);


#include "INCLUDE/BIT.h"
#include "INCLUDE/Delaunay.h"
#include "INCLUDE/GainType.h"
#include "INCLUDE/Genetic.h"
#include "INCLUDE/GeoConversion.h"
#include "INCLUDE/Hashing.h"
#include "INCLUDE/Heap.h"
#include "INCLUDE/LKH.h"
#include "INCLUDE/Segment.h"
#include "INCLUDE/Sequence.h"
#include "INCLUDE/gpx.h"
