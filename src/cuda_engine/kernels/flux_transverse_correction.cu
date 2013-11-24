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
__global__ void correctTransverseFluxGradients_dev(Cons1D *Ul_x1Face_dev, Cons1D *Ur_x1Face_dev, Cons1D *x2Flux_dev, int is, int ie, int js, int je, int sizex, Real hdtodx2, Real hdtodx1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  __shared__ Cons1D x2Flux_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D x2Flux_dev_shared_1[BLOCK_SIZE];
  __shared__ Cons1D x2Flux_dev_shared_2[BLOCK_SIZE];
  __shared__ Cons1D x2Flux_dev_shared_3[BLOCK_SIZE];
  __shared__ Cons1D Ul_x1Face_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D Ur_x1Face_dev_shared[BLOCK_SIZE];

  x2Flux_dev_shared[threadIdx.x] = x2Flux_dev[ind+sizex-1];
  x2Flux_dev_shared_1[threadIdx.x] = x2Flux_dev[ind-1];
  x2Flux_dev_shared_2[threadIdx.x] = x2Flux_dev[ind+sizex];
  x2Flux_dev_shared_3[threadIdx.x] = x2Flux_dev[ind];
  Ul_x1Face_dev_shared[threadIdx.x] = Ul_x1Face_dev[ind];
  Ur_x1Face_dev_shared[threadIdx.x] = Ur_x1Face_dev[ind];

  Ul_x1Face_dev_shared[threadIdx.x].d  -= hdtodx2*(x2Flux_dev_shared[threadIdx.x].d  - x2Flux_dev_shared_1[threadIdx.x].d );
  Ul_x1Face_dev_shared[threadIdx.x].Mx -= hdtodx2*(x2Flux_dev_shared[threadIdx.x].Mz - x2Flux_dev_shared_1[threadIdx.x].Mz);
  Ul_x1Face_dev_shared[threadIdx.x].My -= hdtodx2*(x2Flux_dev_shared[threadIdx.x].Mx - x2Flux_dev_shared_1[threadIdx.x].Mx);
  Ul_x1Face_dev_shared[threadIdx.x].Mz -= hdtodx2*(x2Flux_dev_shared[threadIdx.x].My - x2Flux_dev_shared_1[threadIdx.x].My);
  Ul_x1Face_dev_shared[threadIdx.x].E  -= hdtodx2*(x2Flux_dev_shared[threadIdx.x].E  - x2Flux_dev_shared_1[threadIdx.x].E );
  Ul_x1Face_dev_shared[threadIdx.x].Bz -= hdtodx2*(x2Flux_dev_shared[threadIdx.x].By - x2Flux_dev_shared_1[threadIdx.x].By);

  Ur_x1Face_dev_shared[threadIdx.x].d  -= hdtodx2*(x2Flux_dev_shared_2[threadIdx.x].d  - x2Flux_dev_shared_3[threadIdx.x].d );
  Ur_x1Face_dev_shared[threadIdx.x].Mx -= hdtodx2*(x2Flux_dev_shared_2[threadIdx.x].Mz - x2Flux_dev_shared_3[threadIdx.x].Mz);
  Ur_x1Face_dev_shared[threadIdx.x].My -= hdtodx2*(x2Flux_dev_shared_2[threadIdx.x].Mx - x2Flux_dev_shared_3[threadIdx.x].Mx);
  Ur_x1Face_dev_shared[threadIdx.x].Mz -= hdtodx2*(x2Flux_dev_shared_2[threadIdx.x].My - x2Flux_dev_shared_3[threadIdx.x].My);
  Ur_x1Face_dev_shared[threadIdx.x].E  -= hdtodx2*(x2Flux_dev_shared_2[threadIdx.x].E  - x2Flux_dev_shared_3[threadIdx.x].E );
  Ur_x1Face_dev_shared[threadIdx.x].Bz -= hdtodx2*(x2Flux_dev_shared_2[threadIdx.x].By - x2Flux_dev_shared_3[threadIdx.x].By);

  Ul_x1Face_dev[ind] = Ul_x1Face_dev_shared[threadIdx.x];
  Ur_x1Face_dev[ind] = Ur_x1Face_dev_shared[threadIdx.x];

}

// Started as:
// correctTransverseFluxGradients_A_dev<<<nnBlocks, BLOCK_SIZE>>>(Ul_x2Face_dev, Ur_x2Face_dev, x1Flux_dev, is-1, ie+1, js-1, ju, sizex, hdtodx2, hdtodx1);
/**
 * With shared memory optimization
 */
__global__ void correctTransverseFluxGradients_A_dev(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Cons1D *x1Flux_dev, int is, int ie, int js, int je, int sizex, Real hdtodx2, Real hdtodx1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  __shared__ Cons1D Ul_x2Face_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D Ur_x2Face_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D x1Flux_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D x1Flux_dev_shared_1[BLOCK_SIZE];
  __shared__ Cons1D x1Flux_dev_shared_2[BLOCK_SIZE];
  __shared__ Cons1D x1Flux_dev_shared_3[BLOCK_SIZE];

  Ul_x2Face_dev_shared[threadIdx.x] = Ul_x2Face_dev[ind];
  Ur_x2Face_dev_shared[threadIdx.x] = Ur_x2Face_dev[ind];
  x1Flux_dev_shared[threadIdx.x] = x1Flux_dev[ind-sizex+1];
  x1Flux_dev_shared_1[threadIdx.x] = x1Flux_dev[ind-sizex];
  x1Flux_dev_shared_2[threadIdx.x] = x1Flux_dev[ind+1];
  x1Flux_dev_shared_3[threadIdx.x] = x1Flux_dev[ind];

  Ul_x2Face_dev_shared[threadIdx.x].d  -= hdtodx1*(x1Flux_dev_shared[threadIdx.x].d  - x1Flux_dev_shared_1[threadIdx.x].d );
  Ul_x2Face_dev_shared[threadIdx.x].Mx -= hdtodx1*(x1Flux_dev_shared[threadIdx.x].My - x1Flux_dev_shared_1[threadIdx.x].My);
  Ul_x2Face_dev_shared[threadIdx.x].My -= hdtodx1*(x1Flux_dev_shared[threadIdx.x].Mz - x1Flux_dev_shared_1[threadIdx.x].Mz);
  Ul_x2Face_dev_shared[threadIdx.x].Mz -= hdtodx1*(x1Flux_dev_shared[threadIdx.x].Mx - x1Flux_dev_shared_1[threadIdx.x].Mx);
  Ul_x2Face_dev_shared[threadIdx.x].E  -= hdtodx1*(x1Flux_dev_shared[threadIdx.x].E  - x1Flux_dev_shared_1[threadIdx.x].E );
  Ul_x2Face_dev_shared[threadIdx.x].By -= hdtodx1*(x1Flux_dev_shared[threadIdx.x].Bz - x1Flux_dev_shared_1[threadIdx.x].Bz);

  Ur_x2Face_dev_shared[threadIdx.x].d  -= hdtodx1*(x1Flux_dev_shared_2[threadIdx.x].d  - x1Flux_dev_shared_3[threadIdx.x].d );
  Ur_x2Face_dev_shared[threadIdx.x].Mx -= hdtodx1*(x1Flux_dev_shared_2[threadIdx.x].My - x1Flux_dev_shared_3[threadIdx.x].My);
  Ur_x2Face_dev_shared[threadIdx.x].My -= hdtodx1*(x1Flux_dev_shared_2[threadIdx.x].Mz - x1Flux_dev_shared_3[threadIdx.x].Mz);
  Ur_x2Face_dev_shared[threadIdx.x].Mz -= hdtodx1*(x1Flux_dev_shared_2[threadIdx.x].Mx - x1Flux_dev_shared_3[threadIdx.x].Mx);
  Ur_x2Face_dev_shared[threadIdx.x].E  -= hdtodx1*(x1Flux_dev_shared_2[threadIdx.x].E  - x1Flux_dev_shared_3[threadIdx.x].E );
  Ur_x2Face_dev_shared[threadIdx.x].By  -= hdtodx1*(x1Flux_dev_shared_2[threadIdx.x].Bz  - x1Flux_dev_shared_3[threadIdx.x].Bz );

  Ul_x2Face_dev[ind] = Ul_x2Face_dev_shared[threadIdx.x];
  Ur_x2Face_dev[ind] = Ur_x2Face_dev_shared[threadIdx.x];

}
