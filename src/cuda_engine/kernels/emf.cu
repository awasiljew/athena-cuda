//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

/**
 * With shared memory optimization
 */
__global__ void emf_3_dev(Real *emf3_cc_dev, Gas *U, int is, int ie, int js, int je, int sizex) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  __shared__ Gas U_shared[BLOCK_SIZE];
  U_shared[threadIdx.x] = U[ind];

  emf3_cc_dev[j*sizex+i] =
    (U_shared[threadIdx.x].B1c*U_shared[threadIdx.x].M2 -
     U_shared[threadIdx.x].B2c*U_shared[threadIdx.x].M1 )/U_shared[threadIdx.x].d;

}

// Started as:
// integrate_emf3_corner_dev<<<nnBlocks, BLOCK_SIZE>>>(x1Flux_dev, x2Flux_dev, emf3_dev, emf3_cc_dev, pG, is-1, ie+2, js-1, je+2, sizex);
__global__ void integrate_emf3_corner_dev(Cons1D *x1Flux_dev, Cons1D *x2Flux_dev, Real *emf3_dev, Real *emf3_cc_dev, int is, int ie, int js, int je, int sizex)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  Real emf_l1, emf_r1, emf_l2, emf_r2;

  int ind = j*sizex+i;

  //x1Flux_dev[ind-sizex] 9 times
  //x1Flux_dev[ind] 9 times

  //x2Flux_dev[ind-1] 9 times
  //x2Flux_dev[ind] 9 times

  //emf3_cc_dev[ind-sizex-1] 4 times
  //emf3_cc_dev[ind-sizex] 4 times
  //emf3_cc_dev[ind-1] 4 times
  //emf3_cc_dev[ind] 4 times


  if (x1Flux_dev[ind-sizex].d > 0.0) {
	emf_l2 = -x1Flux_dev[ind-sizex].By
	  + (x2Flux_dev[ind-1].Bz - emf3_cc_dev[ind-sizex-1]);
  }
  else if (x1Flux_dev[ind-sizex].d < 0.0) {
	emf_l2 = -x1Flux_dev[ind-sizex].By
	  + (x2Flux_dev[ind].Bz - emf3_cc_dev[ind-sizex]);
  } else {
	emf_l2 = -x1Flux_dev[ind-sizex].By
	  + 0.5*(x2Flux_dev[ind-1].Bz - emf3_cc_dev[ind-sizex-1] +
		 x2Flux_dev[ind].Bz - emf3_cc_dev[ind-sizex] );
  }

//  if (x1Flux[j-1][i].d > 0.0) {
//      	emf_l2 = -x1Flux[j-1][i].By
//      	  + (x2Flux[j][i-1].Bz - emf3_cc[j-1][i-1]);
//            }
//            else if (x1Flux[j-1][i].d < 0.0) {
//      	emf_l2 = -x1Flux[j-1][i].By
//      	  + (x2Flux[j][i].Bz - emf3_cc[j-1][i]);
//            } else {
//      	emf_l2 = -x1Flux[j-1][i].By
//      	  + 0.5*(x2Flux[j][i-1].Bz - emf3_cc[j-1][i-1] +
//      		 x2Flux[j][i  ].Bz - emf3_cc[j-1][i  ] );
//            }
//

  if (x1Flux_dev[ind].d > 0.0) {
	emf_r2 = -x1Flux_dev[ind].By
	  + (x2Flux_dev[ind-1].Bz - emf3_cc_dev[ind-1]);
  }
  else if (x1Flux_dev[ind].d < 0.0) {
	emf_r2 = -x1Flux_dev[ind].By
	  + (x2Flux_dev[ind].Bz - emf3_cc_dev[ind]);
  }
  else {
	emf_r2 = -x1Flux_dev[ind].By
	  + 0.5*(x2Flux_dev[ind-1].Bz - emf3_cc_dev[ind-1] +
		 x2Flux_dev[ind].Bz - emf3_cc_dev[ind] );
  }

      //            if (x1Flux[j][i].d > 0.0) {
      //      	emf_r2 = -x1Flux[j][i].By
      //      	  + (x2Flux[j][i-1].Bz - emf3_cc[j][i-1]);
      //            }
      //            else if (x1Flux[j][i].d < 0.0) {
      //      	emf_r2 = -x1Flux[j][i].By
      //      	  + (x2Flux[j][i].Bz - emf3_cc[j][i]);
      //
      //            } else {
      //      	emf_r2 = -x1Flux[j][i].By
      //      	  + 0.5*(x2Flux[j][i-1].Bz - emf3_cc[j][i-1] +
      //      		 x2Flux[j][i  ].Bz - emf3_cc[j][i  ] );
      //            }


  if (x2Flux_dev[ind-1].d > 0.0) {
	emf_l1 = x2Flux_dev[ind-1].Bz
	  + (-x1Flux_dev[ind-sizex].By - emf3_cc_dev[ind-sizex-1]);
  }
  else if (x2Flux_dev[ind-1].d < 0.0) {
	emf_l1 = x2Flux_dev[ind-1].Bz
          + (-x1Flux_dev[ind].By - emf3_cc_dev[ind-1]);
  } else {
	emf_l1 = x2Flux_dev[ind-1].Bz
	  + 0.5*(-x1Flux_dev[ind-sizex].By - emf3_cc_dev[ind-sizex-1]
		 -x1Flux_dev[ind].By - emf3_cc_dev[ind-1] );
  }

      //            if (x2Flux[j][i-1].d > 0.0) {
      //      	emf_l1 = x2Flux[j][i-1].Bz
      //      	  + (-x1Flux[j-1][i].By - emf3_cc[j-1][i-1]);
      //            }
      //            else if (x2Flux[j][i-1].d < 0.0) {
      //      	emf_l1 = x2Flux[j][i-1].Bz
      //                + (-x1Flux[j][i].By - emf3_cc[j][i-1]);
      //            } else {
      //      	emf_l1 = x2Flux[j][i-1].Bz
      //      	  + 0.5*(-x1Flux[j-1][i].By - emf3_cc[j-1][i-1]
      //      		 -x1Flux[j  ][i].By - emf3_cc[j  ][i-1] );
      //            }

  if (x2Flux_dev[ind].d > 0.0) {
	emf_r1 = x2Flux_dev[ind].Bz
	  + (-x1Flux_dev[ind-sizex].By - emf3_cc_dev[ind-sizex]);
      }
  else if (x2Flux_dev[ind].d < 0.0) {
	emf_r1 = x2Flux_dev[ind].Bz
	  + (-x1Flux_dev[ind].By - emf3_cc_dev[ind]);
  } else {
	emf_r1 = x2Flux_dev[ind].Bz
	  + 0.5*(-x1Flux_dev[ind-sizex].By - emf3_cc_dev[ind-sizex]
		 -x1Flux_dev[ind].By - emf3_cc_dev[ind] );
  }

      //            if (x2Flux[j][i].d > 0.0) {
      //      	emf_r1 = x2Flux[j][i].Bz
      //      	  + (-x1Flux[j-1][i].By - emf3_cc[j-1][i]);
      //            }
      //            else if (x2Flux[j][i].d < 0.0) {
      //      	emf_r1 = x2Flux[j][i].Bz
      //      	  + (-x1Flux[j][i].By - emf3_cc[j][i]);
      //            } else {
      //      	emf_r1 = x2Flux[j][i].Bz
      //      	  + 0.5*(-x1Flux[j-1][i].By - emf3_cc[j-1][i]
      //      		 -x1Flux[j  ][i].By - emf3_cc[j  ][i] );
      //            }

  emf3_dev[ind] = 0.25*(emf_l1 + emf_r1 + emf_l2 + emf_r2);

      //            emf3[j][i] = 0.25*(emf_l1 + emf_r1 + emf_l2 + emf_r2);
}

// Started as:
// cc_emf3_dev<<<nnBlocks, BLOCK_SIZE>>>(dhalf_dev, x1Flux_dev, x2Flux_dev, B1_x1Face_dev, B2_x2Face_dev, emf3_cc_dev, pG, pG->U, is-1, ie+1, js-1, je+1, sizex, hdtodx1, hdtodx2);
/**
 * With shared optimization
 */
__global__ void cc_emf3_dev(Real* dhalf_dev, Cons1D *x1Flux_dev, Cons1D *x2Flux_dev, Real *B1_x1Face_dev, Real *B2_x2Face_dev, Real *emf3_cc_dev, Grid_gpu *pG, Gas *U, int is, int ie, int js, int je, int sizex, Real hdtodx1, Real hdtodx2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  __shared__ Real d[BLOCK_SIZE];
  __shared__ Real M1[BLOCK_SIZE];
  __shared__ Real M2[BLOCK_SIZE];
  __shared__ Real B1c[BLOCK_SIZE];
  __shared__ Real B2c[BLOCK_SIZE];
  __shared__ Cons1D x1Flux_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D x1Flux_dev_shared_1[BLOCK_SIZE];
  __shared__ Cons1D x2Flux_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D x2Flux_dev_shared_1[BLOCK_SIZE];
  __shared__ Gas U_shared[BLOCK_SIZE];

  x1Flux_dev_shared[threadIdx.x] = x1Flux_dev[ind];
  x1Flux_dev_shared_1[threadIdx.x] = x1Flux_dev[ind+1];
  x2Flux_dev_shared[threadIdx.x] = x2Flux_dev[ind];
  x2Flux_dev_shared_1[threadIdx.x] = x2Flux_dev[ind+sizex];
  U_shared[threadIdx.x] = U[ind];

  d[threadIdx.x] = dhalf_dev[ind];

  M1[threadIdx.x] = U_shared[threadIdx.x].M1
        - hdtodx1*(x1Flux_dev_shared_1[threadIdx.x].Mx - x1Flux_dev_shared[threadIdx.x].Mx)
        - hdtodx2*(x2Flux_dev_shared_1[threadIdx.x].Mz - x2Flux_dev_shared[threadIdx.x].Mz);

  M2[threadIdx.x] = U_shared[threadIdx.x].M2
        - hdtodx1*(x1Flux_dev_shared_1[threadIdx.x].My - x1Flux_dev_shared[threadIdx.x].My)
        - hdtodx2*(x2Flux_dev_shared_1[threadIdx.x].Mx - x2Flux_dev_shared[threadIdx.x].Mx);

  B1c[threadIdx.x] = 0.5*(B1_x1Face_dev[ind] + B1_x1Face_dev[ind+1]);
  B2c[threadIdx.x] = 0.5*(B2_x2Face_dev[ind] + B2_x2Face_dev[ind+sizex]);

  emf3_cc_dev[ind] = (B1c[threadIdx.x]*M2[threadIdx.x] - B2c[threadIdx.x]*M1[threadIdx.x])/d[threadIdx.x];

}

