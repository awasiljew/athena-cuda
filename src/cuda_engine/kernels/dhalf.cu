//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

// Started as:
// dhalf_init_dev<<<nnBlocks, BLOCK_SIZE>>>(dhalf_dev, x1Flux_dev, x2Flux_dev, pG->U, is-1, ie+1, js-1, je+1, sizex, hdtodx1, hdtodx2);
__global__ void dhalf_init_dev(Real *dhalf_dev, Cons1D *x1Flux_dev, Cons1D *x2Flux_dev, Gas *U, int is, int ie, int js, int je, int sizex, Real hdtodx1, Real hdtodx2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  dhalf_dev[ind] = U[ind].d
          - hdtodx1*(x1Flux_dev[ind+1].d - x1Flux_dev[ind].d)
          - hdtodx2*(x2Flux_dev[ind+sizex].d - x2Flux_dev[ind].d);

//  for (j=js-1; j<=je+1; j++) {
//        for (i=is-1; i<=ie+1; i++) {
//          dhalf[j][i] = pG->U[j][i].d
//            - hdtodx1*(x1Flux[j  ][i+1].d - x1Flux[j][i].d)
//            - hdtodx2*(x2Flux[j+1][i  ].d - x2Flux[j][i].d);
//        }
//      }

}
