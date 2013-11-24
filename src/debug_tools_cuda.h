//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

#ifndef DEBUG_TOOLS_CUDA_H_
#define DEBUG_TOOLS_CUDA_H_

#include "defs.h"
#include "athena.h"

//#define REALS_CMP_DETAILS
//#define REALS_CMP_ARR
#define GAS_CMP_DETAILS
#define GAS_CMP_DETAILS_ARR
#define PRIM1D_CMP_DETAILS
#define PRIM1D_CMP_DETAILS_ARR
#define CONS1D_CMP_DETAILS
#define CONS1D_CMP_DETAILS_ARR
#define GRID_CMP_DETAILS_U
#define GRID_CMP_DETAILS_B1I
#define GRID_CMP_DETAILS_B2I
#define GRID_CMP_DETAILS_B3I

//#define DEBUG_STEP_0
//#define DEBUG_STEP_1a
//#define DEBUG_STEP_1b
//#define DEBUG_STEP_1c
//#define DEBUG_STEP_1d
//#define DEBUG_STEP_1
//#define DEBUG_STEP_2b
//#define DEBUG_STEP_2c
//#define DEBUG_STEP_2
//#define DEBUG_STEP_3
//#define DEBUG_STEP_4a
//#define DEBUG_STEP_4
//#define DEBUG_STEP_5
//#define DEBUG_STEP_6
//#define DEBUG_STEP_7
//#define DEBUG_STEP_8b
//#define DEBUG_STEP_8
//#define DEBUG_STEP_9
//#define DEBUG_STEP_11

//#define ESYS_ROE_CUDA
//#define STEP_1a_CUDA
//#define STEP_2a_CUDA
//#define STEP2b_CUDA
//#define STEP_3_CUDA
//#define STEP_4a_CUDA
//#define STEP_4b_CUDA
//#define STEP_5a_CUDA
//#define STEP_5b_CUDA
//#define STEP_6a_CUDA
//#define STEP_6b_CUDA
//#define STEP_7_CUDA
//#define STEP_8b_CUDA
//#define STEP_8c_CUDA
//#define STEP_9_CUDA
//#define STEP_9b_CUDA
//#define STEP_11a_CUDA
//#define STEP_11b_CUDA
//#define STEP_13_CUDA

//#define BIG_STEP_1_CUDA
//#define BIG_STEP_2_CUDA
//#define ONLY_HOST
//#define GPU_HOST_COMPARE

#define FIELD_LOOP
//#define BLAST_PROBLEM
//#define CPAW2D

#define ACCURACY 0.0

extern Cons1D **x1Flux;
extern Cons1D **x2Flux;
extern Cons1D **U1d;
extern Prim1D **W, **Wl, **Wr;
extern Cons1D **Ul_x1Face, **Ur_x1Face;
extern Real **Bxc, **Bxi;
extern Real **emf3, **emf3_cc;
extern Real **B1_x1Face, **B2_x2Face;
extern Cons1D **Ul_x2Face, **Ur_x2Face;

extern int ESYS_ROE_CUDA_FLAG;

int compare_reals(Real r1, Real r2);
int compare_reals_array(Real *r1, Real* r2, int n, int lo, int hi);
int compare_gas(Gas g1, Gas g2);
int compare_gas_array(Gas* g1, Gas *g2, int n, int lo, int hi);
int compare_Prim1D(Prim1D p1, Prim1D p2);
int compare_Prim1D_array(Prim1D* p1, Prim1D* p2, int n, int lo, int hi);
int compare_Cons1D(Cons1D c1, Cons1D c2);
int compare_Cons1D_array(Cons1D* c1, Cons1D* c2, int n, int lo, int hi);
int compare_grid_cpu(Grid* g1, Grid* g2, int ghost);
int compare_grid_gpu(Grid_gpu* g1, Grid* g2, Grid* workGrid, int ghost);

#endif /*DEBUG_TOOLS_CUDA_H_ */
