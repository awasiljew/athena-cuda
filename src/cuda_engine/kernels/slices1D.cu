//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif


/**
 * Load data for U1d and Bxc. Run over whole domain.
 */
__global__ void load_1a_dev(Cons1D *U1d_dev, Real *Bxc_dev, Gas *U, int is, int ie, int js, int je, int sizex) {
	/* Calculate index */
	int i, j;
	calculateIndexes2D(&i, &j, sizex);

	/* Check bounds */
	if(i < is || i > ie || j < js || j > je) return;

	int ind = j*sizex+i;

	U1d_dev[ind].d = U[ind].d;
	U1d_dev[ind].Mx = U[ind].M1;
	U1d_dev[ind].My = U[ind].M2;
	U1d_dev[ind].Mz = U[ind].M3;
	U1d_dev[ind].E  = U[ind].E;
	U1d_dev[ind].By = U[ind].B2c;
	U1d_dev[ind].Bz = U[ind].B3c;
	Bxc_dev[ind] = U[ind].B1c;

}

/**
 * Load data for U1d and Bxc. Run over whole domain.
 */
__global__ void load_2a_dev(Cons1D *U1d_dev, Real *Bxc_dev, Gas *U, int is, int ie, int js, int je, int sizex, int sizey) {
	/* Calculate index */
	int i, j/*, ib, jb*/;
	calculateIndexes2D(&i, &j, sizex);

	/* Check bounds */
	if(i < is || i > ie || j < js || j > je) return;

	int ind = j*sizex+i;

	U1d_dev[ind].d = U[ind].d;
	U1d_dev[ind].Mx = U[ind].M2;
	U1d_dev[ind].My = U[ind].M3;
	U1d_dev[ind].Mz = U[ind].M1;
	U1d_dev[ind].E  = U[ind].E;
	U1d_dev[ind].By = U[ind].B3c;
	U1d_dev[ind].Bz = U[ind].B1c;
	Bxc_dev[i*sizex+j] = U[ind].B2c;
}


// Started as:
// load1DSlice_dev<<<nBlocks,BLOCK_SIZE>>>(U1d_dev, Bxc_dev, Bxi_dev, B1_x1Face_dev, pG->U, pG->B1i, j, is-nghost, ie+nghost, sizex);
__global__ void load1DSlice_dev(Cons1D *U1d_dev, Real *Bxc_dev, Real *Bxi_dev, Real *B1_x1Face_dev, Gas *U, Real *B1i, int j, int is, int ie, int sizex) {
  /* Calculate and check index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < is || i > ie) return;
  int ind = j*sizex+i;

  U1d_dev[i].d = U[ind].d;
  U1d_dev[i].Mx = U[ind].M1;
  U1d_dev[i].My = U[ind].M2;
  U1d_dev[i].Mz = U[ind].M3;
  U1d_dev[i].E  = U[ind].E;
  U1d_dev[i].By = U[ind].B2c;
  U1d_dev[i].Bz = U[ind].B3c;
  Bxc_dev[i] = U[ind].B1c;
  Bxi_dev[i] = B1i[ind];
  B1_x1Face_dev[ind] = B1i[ind];

////////////////////////////////////////////////////////
//  for (j=jl; j<=ju; j++) {
//      for (i=is-nghost; i<=ie+nghost; i++) {
//        U1d[i].d  = pG->U[j][i].d;
//        U1d[i].Mx = pG->U[j][i].M1;
//        U1d[i].My = pG->U[j][i].M2;
//        U1d[i].Mz = pG->U[j][i].M3;
//        U1d[i].E  = pG->U[j][i].E;
//        U1d[i].By = pG->U[j][i].B2c;
//        U1d[i].Bz = pG->U[j][i].B3c;
//        Bxc[i] = pG->U[j][i].B1c;
//        Bxi[i] = pG->B1i[j][i];
//        B1_x1Face[j][i] = pG->B1i[j][i];
//      }
}

// Started as:
// load1DSlice_y_dev<<<nBlocks_y,BLOCK_SIZE>>>(U1d_dev, Bxc_dev, Bxi_dev, B2_x2Face_dev, pG->U, pG->B2i, i, js-nghost, je+nghost, sizex);
__global__ void load1DSlice_y_dev(Cons1D *U1d_dev, Real *Bxc_dev, Real *Bxi_dev, Real *B2_x2Face_dev, Gas *U, Real *B2i, int i, int js, int je, int sizex) {
  /* Calculate and check index */
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(j < js || j > je) return;

  int ind = j*sizex+i;

  U1d_dev[j].d  = U[ind].d;
  U1d_dev[j].Mx = U[ind].M2;
  U1d_dev[j].My = U[ind].M3;
  U1d_dev[j].Mz = U[ind].M1;
  U1d_dev[j].E  = U[ind].E;
  U1d_dev[j].By = U[ind].B3c;
  U1d_dev[j].Bz = U[ind].B1c;
  Bxc_dev[j] = U[ind].B2c;
  Bxi_dev[j] = B2i[ind];
  B2_x2Face_dev[ind] = B2i[ind];

//  for (i=il; i<=iu; i++) {
//      for (j=js-nghost; j<=je+nghost; j++) {
//        U1d[j].d  = pG->U[j][i].d;
//        U1d[j].Mx = pG->U[j][i].M2;
//        U1d[j].My = pG->U[j][i].M3;
//        U1d[j].Mz = pG->U[j][i].M1;
//        U1d[j].E  = pG->U[j][i].E;
//        U1d[j].By = pG->U[j][i].B3c;
//        U1d[j].Bz = pG->U[j][i].B1c;
//        Bxc[j] = pG->U[j][i].B2c;
//        Bxi[j] = pG->B2i[j][i];
//        B2_x2Face[j][i] = pG->B2i[j][i];
//      }
//  }

}

// Started as
// Cons1D_to_Prim1DSlice_dev<<<nBlocks, BLOCK_SIZE>>>(U1d_dev, W_dev, Bxc_dev, is-nghost, ie+nghost, Gamma_1);
// Cons1D_to_Prim1DSlice_dev<<<nBlocks_y, BLOCK_SIZE>>>(U1d_dev, W_dev, Bxc_dev, js-nghost, je+nghost, Gamma_1);
__global__ void Cons1D_to_Prim1DSlice_dev(Cons1D *pU, Prim1D *pW, Real *pBx, int is, int ie, Real Gamma_1) {
  /* Calculate and check index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < is || i > ie) return;

  Cons1D_to_Prim1D_cu_dev(&pU[i],&pW[i],&pBx[i], Gamma_1);

//  pU[i].d  = pW[i].d;
//  pU[i].Mx = pW[i].d*pW[i].Vx;
//  pU[i].My = pW[i].d*pW[i].Vy;
//  pU[i].Mz = pW[i].d*pW[i].Vz;
//
//  pU[i].E = pW[i].P/Gamma_1 + 0.5*pW->d*(SQR(pW[i].Vx) + SQR(pW[i].Vy) + SQR(pW[i].Vz));
//  pU[i].E += 0.5*(SQR(pBx[i]) + SQR(pW[i].By) + SQR(pW[i].Bz));
//
//  pU[i].By = pW[i].By;
//  pU[i].Bz = pW[i].Bz;


////////////////////////////////////////////////////////////////
//  for (j=js-nghost; j<=je+nghost; j++) {
//        Cons1D_to_Prim1D(&U1d[j],&W[j],&Bxc[j]);
//      }
}

/**
 * Run over 1D slice
 */
__global__ void Cons1D_to_Prim1D_1b_dev(Cons1D *pU, Prim1D *pW, Real *pBx, int is, int ie, int j, int sizex, Real Gamma_1) {
  /* Calculate and check index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < is || i > ie) return;
  int ind = j*sizex+i;

  Cons1D_to_Prim1D_cu_dev(&pU[ind],&pW[ind],&pBx[ind], Gamma_1);

}

/**
 * Run over 1D slice
 */
__global__ void Cons1D_to_Prim1D_2b_dev(Cons1D *pU, Prim1D *pW, Real *pBx, int js, int je, int i, int sizex, int sizey, Real Gamma_1) {
  /* Calculate and check index */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < js || j > je) return;
  int ind = j*sizex+i;

  Cons1D_to_Prim1D_cu_dev(&pU[ind],&pW[i*sizex+j],&pBx[i*sizex+j], Gamma_1);

}

// Started as:
// Prim1D_to_Cons1DSlice_dev<<<nBlocks, BLOCK_SIZE>>>(Ul_x1Face_dev, Wl_dev, Bxi_dev, is-1, iu, j, sizex, Gamma_1);
__global__ void Prim1D_to_Cons1DSlice_dev(Cons1D *pU, Prim1D *pW, Real *pBx, int is, int ie, int j, int sizex, Real Gamma_1) {
  /* Calculate and check index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < is || i > ie) return;

  Prim1D_to_Cons1D_cu_dev(&pU[j*sizex+i],&pW[i],&pBx[i], Gamma_1);
}

// Started as:
// Prim1D_to_Cons1DSlice_y_dev<<<nBlocks_y, BLOCK_SIZE>>>(Ul_x2Face_dev, Wl_dev, Bxi_dev, js-1, ju, i, sizex, Gamma_1);
// Prim1D_to_Cons1DSlice_y_dev<<<nBlocks_y, BLOCK_SIZE>>>(Ur_x2Face_dev, Wr_dev, Bxi_dev, js-1, ju, i, sizex, Gamma_1);
__global__ void Prim1D_to_Cons1DSlice_y_dev(Cons1D *pU, Prim1D *pW, Real *pBx, int js, int je, int i, int sizex, Real Gamma_1) {
  /* Calculate and check index */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < js || j > je) return;

  Prim1D_to_Cons1D_cu_dev(&pU[j*sizex+i],&pW[j],&pBx[j], Gamma_1);

#ifdef __DEVICE_EMULATION__

  /* Check */
//  if(Wl_shared[threadIdx.x].By != 0.0f && fabs(Wl_shared[threadIdx.x].By) < 1.0e-22 ) {
//	  printf("BY LESS THAN TINY NUMBER!!! %d %e\n", i+1, Wl_shared[threadIdx.x].By);
  //}

#endif


// for (i=il; i<=iu; i++) {
//  for (j=js-1; j<=ju; j++) {
//        Prim1D_to_Cons1D(&Ul_x2Face[j][i],&Wl[j],&Bxi[j]);
//        Prim1D_to_Cons1D(&Ur_x2Face[j][i],&Wr[j],&Bxi[j]);
//
//        flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[j],Wr[j],
//                   B2_x2Face[j][i],&x2Flux[j][i]);
//      }
// }
}

__global__ void Cons1D_to_Prim1D_Slice1D_1e(Cons1D *Ul_x1Face_dev, Cons1D *Ur_x1Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B1_x1Face_dev, Cons1D *x1Flux_dev, int is, int ie, int js, int je, int sizex, Real Gamma_1, Real Gamma_2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j;
  calculateIndexes2D(&i, &j, sizex);

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  /* Main algorithm */
  Prim1D_to_Cons1D_cu_dev(&Ul_x1Face_dev[ind],&Wl_dev[ind],&B1_x1Face_dev[ind], Gamma_1);
  Prim1D_to_Cons1D_cu_dev(&Ur_x1Face_dev[ind],&Wr_dev[ind],&B1_x1Face_dev[ind], Gamma_1);

  flux_roe_cu_dev(Ul_x1Face_dev[ind],Ur_x1Face_dev[ind],Wl_dev[ind],Wr_dev[ind],
                 B1_x1Face_dev[ind],&x1Flux_dev[ind], Gamma_1, Gamma_2);
}

__global__ void Cons1D_to_Prim1D_Slice1D_2e(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B2_x2Face_dev, Cons1D *x2Flux_dev, int is, int ie, int js, int je, int sizex, int sizey, Real Gamma_1, Real Gamma_2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j;
  calculateIndexes2D(&i, &j, sizex);

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;
  int ind2 = i*sizey+j;

  /* Main algorithm */
  Prim1D_to_Cons1D_cu_dev(&Ul_x2Face_dev[ind],&Wl_dev[ind2],&B2_x2Face_dev[ind], Gamma_1);
  Prim1D_to_Cons1D_cu_dev(&Ur_x2Face_dev[ind],&Wr_dev[ind2],&B2_x2Face_dev[ind], Gamma_1);

  flux_roe_cu_dev(Ul_x2Face_dev[ind],Ur_x2Face_dev[ind],Wl_dev[ind2],Wr_dev[ind2],
          B2_x2Face_dev[ind],x2Flux_dev+ind, Gamma_1, Gamma_2);
}
