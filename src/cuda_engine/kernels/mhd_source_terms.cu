//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

// Started as:
// addMHDSourceTerms_dev_half<<<nBlocks, BLOCK_SIZE>>>(Wl_dev, Wr_dev, pG->U, pG->B1i, pG->dx1, is-1, iu, j, sizex, hdt);
__global__ void addMHDSourceTerms_dev_half(Prim1D *Wl_dev, Prim1D *Wr_dev, Gas *U, Real *B1i, Real dx1, int is, int ie, int j, int sizex, Real hdt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < is || i > ie) return;
  int ind = j*sizex+i;

  Real MHD_src = (U[ind-1].M2/U[ind-1].d)*
		  (B1i[ind] - B1i[ind-1])/dx1;
  Wl_dev[ind].By += hdt*MHD_src;

  MHD_src = (U[ind].M2/U[ind].d)*
	 (B1i[ind+1] - B1i[ind])/dx1;
  Wr_dev[ind].By += hdt*MHD_src;

///////////////////////////////////////////////////////////////////
//      for (i=is-1; i<=iu; i++) {
//            MHD_src = (pG->U[j][i-1].M2/pG->U[j][i-1].d)*
//                     (pG->B1i[j][i] - pG->B1i[j][i-1])/pG->dx1;
//            Wl[i].By += hdt*MHD_src;
//
//            MHD_src = (pG->U[j][i].M2/pG->U[j][i].d)*
//                     (pG->B1i[j][i+1] - pG->B1i[j][i])/pG->dx1;
//            Wr[i].By += hdt*MHD_src;
//          }
}

// Started as:
// addMHDSourceTerms_y_dev<<<nBlocks_y, BLOCK_SIZE>>>(Wl_dev, Wr_dev, pG->U, pG->B2i, pG->dx2, js-1, ju, i, sizex, hdt);
__global__ void addMHDSourceTerms_dev_half_y(Prim1D *Wl_dev, Prim1D *Wr_dev, Gas *U, Real *B2i, Real dx2, int js, int je, int i, int sizex, int sizey, Real hdt) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < js || j > je) return;
  int ind = i*sizex+j;

    Real MHD_src = (U[j*sizex+i-sizex].M1/U[j*sizex+i-sizex].d)*
               (B2i[j*sizex+i] - B2i[j*sizex+i-sizex])/dx2;
      Wl_dev[ind].Bz += hdt*MHD_src;

      MHD_src = (U[j*sizex+i].M1/U[j*sizex+i].d)*
               (B2i[j*sizex+i+sizex] - B2i[j*sizex+i])/dx2;
      Wr_dev[ind].Bz += hdt*MHD_src;

   //////////

//      MHD_src = (pG->U[j-1][i].M1/pG->U[j-1][i].d)*
//              (pG->B2i[j][i] - pG->B2i[j-1][i])/pG->dx2;
//            Wl[i][j].Bz += hdt*MHD_src;
//
//            MHD_src = (pG->U[j][i].M1/pG->U[j][i].d)*
//              (pG->B2i[j+1][i] - pG->B2i[j][i])/pG->dx2;
//            Wr[i][j].Bz += hdt*MHD_src;

//      /* Add "MHD source terms" for 0.5*dt */
//          for (j=js-1; j<=ju; j++) {
//            MHD_src = (pG->U[j-1][i].M1/pG->U[j-1][i].d)*
//              (pG->B2i[j][i] - pG->B2i[j-1][i])/pG->dx2;
//            Wl[j].Bz += hdt*MHD_src;
//
//            MHD_src = (pG->U[j][i].M1/pG->U[j][i].d)*
//              (pG->B2i[j+1][i] - pG->B2i[j][i])/pG->dx2;
//            Wr[j].Bz += hdt*MHD_src;
//          }

}

/**
 * With shared optimization
 */
__global__ void addMHDSourceTerms_dev(Cons1D *Ul_x1Face_dev, Cons1D *Ur_x1Face_dev, Gas *U, Real *B1i, int is, int ie, int js, int je, int sizex, Real hdtodx2, Real hdtodx1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  __shared__ Real dbx[BLOCK_SIZE];
  __shared__ Real B1[BLOCK_SIZE];
  __shared__ Real B2[BLOCK_SIZE];
  __shared__ Real B3[BLOCK_SIZE];
  __shared__ Real V3[BLOCK_SIZE];
  __shared__ Gas U_shared[BLOCK_SIZE];
  __shared__ Gas U_shared_1[BLOCK_SIZE];
  __shared__ Cons1D Ul_x1Face_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D Ur_x1Face_dev_shared[BLOCK_SIZE];

  U_shared[threadIdx.x] = U[ind-1];
  U_shared_1[threadIdx.x] = U[ind];
  Ul_x1Face_dev_shared[threadIdx.x] = Ul_x1Face_dev[ind];
  Ur_x1Face_dev_shared[threadIdx.x] = Ur_x1Face_dev[ind];

  dbx[threadIdx.x] = B1i[ind] - B1i[ind-1];
  B1[threadIdx.x]  = U_shared[threadIdx.x].B1c;
  B2[threadIdx.x] = U_shared[threadIdx.x].B2c;
  B3[threadIdx.x] = U_shared[threadIdx.x].B3c;
  V3[threadIdx.x] = U_shared[threadIdx.x].M3/U_shared[threadIdx.x].d;

  Ul_x1Face_dev_shared[threadIdx.x].Mx += hdtodx1*B1[threadIdx.x]*dbx[threadIdx.x];
  Ul_x1Face_dev_shared[threadIdx.x].My += hdtodx1*B2[threadIdx.x]*dbx[threadIdx.x];
  Ul_x1Face_dev_shared[threadIdx.x].Mz += hdtodx1*B3[threadIdx.x]*dbx[threadIdx.x];
  Ul_x1Face_dev_shared[threadIdx.x].Bz += hdtodx1*V3[threadIdx.x]*dbx[threadIdx.x];
  Ul_x1Face_dev_shared[threadIdx.x].E  += hdtodx1*B3[threadIdx.x]*V3[threadIdx.x]*dbx[threadIdx.x];

  dbx[threadIdx.x] = B1i[ind+1] - B1i[ind];
  B1[threadIdx.x] = U_shared_1[threadIdx.x].B1c;
  B2[threadIdx.x] = U_shared_1[threadIdx.x].B2c;
  B3[threadIdx.x] = U_shared_1[threadIdx.x].B3c;
  V3[threadIdx.x] = U_shared_1[threadIdx.x].M3/U_shared_1[threadIdx.x].d;

  Ur_x1Face_dev_shared[threadIdx.x].Mx += hdtodx1*B1[threadIdx.x]*dbx[threadIdx.x];
  Ur_x1Face_dev_shared[threadIdx.x].My += hdtodx1*B2[threadIdx.x]*dbx[threadIdx.x];
  Ur_x1Face_dev_shared[threadIdx.x].Mz += hdtodx1*B3[threadIdx.x]*dbx[threadIdx.x];
  Ur_x1Face_dev_shared[threadIdx.x].Bz += hdtodx1*V3[threadIdx.x]*dbx[threadIdx.x];
  Ur_x1Face_dev_shared[threadIdx.x].E  += hdtodx1*B3[threadIdx.x]*V3[threadIdx.x]*dbx[threadIdx.x];

  Ul_x1Face_dev[ind] = Ul_x1Face_dev_shared[threadIdx.x];
  Ur_x1Face_dev[ind] = Ur_x1Face_dev_shared[threadIdx.x];

}

// Started as:
// addMHDSourceTerms_B_dev<<<nnBlocks, BLOCK_SIZE>>>(Ul_x2Face_dev, Ur_x2Face_dev, pG->U, pG->B2i, is-1, ie+1, js-1, ju, sizex, hdtodx2, hdtodx1);
/**
 * With shared optimization
 */
__global__ void addMHDSourceTerms_B_dev(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Gas *U, Real *B2i, int is, int ie, int js, int je, int sizex, Real hdtodx2, Real hdtodx1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  __shared__ Real dby[BLOCK_SIZE];
  __shared__ Real B1[BLOCK_SIZE];
  __shared__ Real B2[BLOCK_SIZE];
  __shared__ Real B3[BLOCK_SIZE];
  __shared__ Real V3[BLOCK_SIZE];
  __shared__ Gas U_shared[BLOCK_SIZE];
  __shared__ Gas U_shared_1[BLOCK_SIZE];
  __shared__ Cons1D Ul_x2Face_dev_shared[BLOCK_SIZE];
  __shared__ Cons1D Ur_x2Face_dev_shared[BLOCK_SIZE];

  U_shared[threadIdx.x] = U[ind-sizex];
  U_shared_1[threadIdx.x] = U[ind];
  Ul_x2Face_dev_shared[threadIdx.x] = Ul_x2Face_dev[ind];
  Ur_x2Face_dev_shared[threadIdx.x] = Ur_x2Face_dev[ind];

  dby[threadIdx.x] = B2i[ind] - B2i[ind-sizex];
  B1[threadIdx.x] = U_shared[threadIdx.x].B1c;
  B2[threadIdx.x] = U_shared[threadIdx.x].B2c;
  B3[threadIdx.x] = U_shared[threadIdx.x].B3c;
  V3[threadIdx.x] = U_shared[threadIdx.x].M3/U_shared[threadIdx.x].d;

  Ul_x2Face_dev_shared[threadIdx.x].Mz += hdtodx2*B1[threadIdx.x]*dby[threadIdx.x];
  Ul_x2Face_dev_shared[threadIdx.x].Mx += hdtodx2*B2[threadIdx.x]*dby[threadIdx.x];
  Ul_x2Face_dev_shared[threadIdx.x].My += hdtodx2*B3[threadIdx.x]*dby[threadIdx.x];
  Ul_x2Face_dev_shared[threadIdx.x].By += hdtodx2*V3[threadIdx.x]*dby[threadIdx.x];
  Ul_x2Face_dev_shared[threadIdx.x].E  += hdtodx2*B3[threadIdx.x]*V3[threadIdx.x]*dby[threadIdx.x];

  dby[threadIdx.x] = B2i[ind+sizex] - B2i[ind];
  B1[threadIdx.x] = U_shared_1[threadIdx.x].B1c;
  B2[threadIdx.x] = U_shared_1[threadIdx.x].B2c;
  B3[threadIdx.x] = U_shared_1[threadIdx.x].B3c;
  V3[threadIdx.x] = U_shared_1[threadIdx.x].M3/U_shared_1[threadIdx.x].d;

  Ur_x2Face_dev_shared[threadIdx.x].Mz += hdtodx2*B1[threadIdx.x]*dby[threadIdx.x];
  Ur_x2Face_dev_shared[threadIdx.x].Mx += hdtodx2*B2[threadIdx.x]*dby[threadIdx.x];
  Ur_x2Face_dev_shared[threadIdx.x].My += hdtodx2*B3[threadIdx.x]*dby[threadIdx.x];
  Ur_x2Face_dev_shared[threadIdx.x].By += hdtodx2*V3[threadIdx.x]*dby[threadIdx.x];
  Ur_x2Face_dev_shared[threadIdx.x].E  += hdtodx2*B3[threadIdx.x]*V3[threadIdx.x]*dby[threadIdx.x];

  Ul_x2Face_dev[ind] = Ul_x2Face_dev_shared[threadIdx.x];
  Ur_x2Face_dev[ind] = Ur_x2Face_dev_shared[threadIdx.x];

//  __shared__ Gas U_shared[BLOCK_SIZE]; // Shared for Gas structure
//  __shared__ Gas U_shared_1[BLOCK_SIZE]; // Shared for Gas structure
//  __shared__ Cons1D Ul_x2Face_dev_shared[BLOCK_SIZE]; // Shared for flux
//  __shared__ Cons1D Ur_x2Face_dev_shared[BLOCK_SIZE]; //Shared for flux
//
//  U_shared[threadIdx.x] = U[ind]; // Load from global
//  U_shared_1[threadIdx.x] = U[ind-sizex]; // Load from global
//  Ul_x2Face_dev_shared[threadIdx.x] = Ul_x2Face_dev[ind]; // Load from global
//  Ur_x2Face_dev_shared[threadIdx.x] = Ur_x2Face_dev[ind]; // Load from global
//
//  Real dby = B2i[ind] - B2i[ind-sizex];
//  Real B1 = U_shared_1[threadIdx.x].B1c;
//  Real B2 = U_shared_1[threadIdx.x].B2c;
//  Real B3 = U_shared_1[threadIdx.x].B3c;
//  Real V3 = U_shared_1[threadIdx.x].M3/U_shared[threadIdx.x].d;
//
//  Ul_x2Face_dev_shared[threadIdx.x].Mz += hdtodx2*B1*dby;
//  Ul_x2Face_dev_shared[threadIdx.x].Mx += hdtodx2*B2*dby;
//  Ul_x2Face_dev_shared[threadIdx.x].My += hdtodx2*B3*dby;
//  Ul_x2Face_dev_shared[threadIdx.x].By += hdtodx2*V3*dby;
//  Ul_x2Face_dev_shared[threadIdx.x].E  += hdtodx2*B3*V3*dby;
//
//  dby = B2i[ind+sizex] - B2i[ind];
//  B1 = U_shared[threadIdx.x].B1c;
//  B2 = U_shared[threadIdx.x].B2c;
//  B3 = U_shared[threadIdx.x].B3c;
//  V3 = U_shared[threadIdx.x].M3/U_shared[threadIdx.x].d;
//
//  Ur_x2Face_dev_shared[threadIdx.x].Mz += hdtodx2*B1*dby;
//  Ur_x2Face_dev_shared[threadIdx.x].Mx += hdtodx2*B2*dby;
//  Ur_x2Face_dev_shared[threadIdx.x].My += hdtodx2*B3*dby;
//  Ur_x2Face_dev_shared[threadIdx.x].By += hdtodx2*V3*dby;
//  Ur_x2Face_dev_shared[threadIdx.x].E  += hdtodx2*B3*V3*dby;
//
//  /* Load back */
//  Ul_x2Face_dev[ind] = Ul_x2Face_dev_shared[threadIdx.x];
//  Ur_x2Face_dev[ind] = Ur_x2Face_dev_shared[threadIdx.x];

//  for (j=js-1; j<=ju; j++) {
//      for (i=is-1; i<=ie+1; i++) {
//        dby = pG->B2i[j][i] - pG->B2i[j-1][i];
//        B1 = pG->U[j-1][i].B1c;
//        B2 = pG->U[j-1][i].B2c;
//        B3 = pG->U[j-1][i].B3c;
//        V3 = pG->U[j-1][i].M3/pG->U[j-1][i].d;
//
//        Ul_x2Face[j][i].Mz += hdtodx2*B1*dby;
//        Ul_x2Face[j][i].Mx += hdtodx2*B2*dby;
//        Ul_x2Face[j][i].My += hdtodx2*B3*dby;
//        Ul_x2Face[j][i].By += hdtodx2*V3*dby;
//        Ul_x2Face[j][i].E  += hdtodx2*B3*V3*dby;
//
//        dby = pG->B2i[j+1][i] - pG->B2i[j][i];
//        B1 = pG->U[j][i].B1c;
//        B2 = pG->U[j][i].B2c;
//        B3 = pG->U[j][i].B3c;
//        V3 = pG->U[j][i].M3/pG->U[j][i].d;
//
//        Ur_x2Face[j][i].Mz += hdtodx2*B1*dby;
//        Ur_x2Face[j][i].Mx += hdtodx2*B2*dby;
//        Ur_x2Face[j][i].My += hdtodx2*B3*dby;
//        Ur_x2Face[j][i].By += hdtodx2*V3*dby;
//        Ur_x2Face[j][i].E  += hdtodx2*B3*V3*dby;
//      }
//    }


}

