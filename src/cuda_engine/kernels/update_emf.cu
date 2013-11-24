//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

// Started as:
// updateMagneticField_4a_dev<<<nnBlocks, BLOCK_SIZE>>>(B1_x1Face_dev, B2_x2Face_dev, emf3_dev, is-1, ie+1, js-1, je+1, sizex, hdtodx2, hdtodx1);
__global__ void updateMagneticField_4a_dev(Real *B1_x1Face_dev, Real *B2_x2Face_dev, Real *emf3_dev, int is, int ie, int js, int je, int sizex, Real hdtodx2, Real hdtodx1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  /* Main algorithm */
  B1_x1Face_dev[ind] -= hdtodx2*(emf3_dev[ind+sizex] - emf3_dev[ind]);
  B2_x2Face_dev[ind] += hdtodx1*(emf3_dev[ind+1] - emf3_dev[ind]);

  //  for (j=js-1; j<=je+1; j++) {
  //      for (i=is-1; i<=ie+1; i++) {
  //  !!!      B1_x1Face[j][i] -= hdtodx2*(emf3[j+1][i  ] - emf3[j][i]);
  //  !!!      B2_x2Face[j][i] += hdtodx1*(emf3[j  ][i+1] - emf3[j][i]);
  //      }
  //      B1_x1Face[j][iu] -= hdtodx2*(emf3[j+1][iu] - emf3[j][iu]);
  //    }
}

// Started as
// updateMagneticField_4b_dev<<<nBlocks_y, BLOCK_SIZE>>>(B1_x1Face_dev, emf3_dev, js-1, je+1, iu, sizex, hdtodx2);
__global__ void updateMagneticField_4b_dev(Real *B1_x1Face_dev, Real *emf3_dev, int js, int je, int iu, int sizex, Real hdtodx2) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  /* Check bounds */
  if(j < js || j > je) return;

  int ind = j*sizex+iu;

  /* Main algorithm */
  B1_x1Face_dev[ind] -= hdtodx2*(emf3_dev[ind+sizex] - emf3_dev[ind]);

//  for (j=js-1; j<=je+1; j++) {
//      for (i=is-1; i<=ie+1; i++) {
//        B1_x1Face[j][i] -= hdtodx2*(emf3[j+1][i  ] - emf3[j][i]);
//        B2_x2Face[j][i] += hdtodx1*(emf3[j  ][i+1] - emf3[j][i]);
//      }
// !!!!!     B1_x1Face[j][iu] -= hdtodx2*(emf3[j+1][iu] - emf3[j][iu]);
//    }
}

// Started as:
// updateMagneticField_4c_dev<<<nBlocks, BLOCK_SIZE>>>(B2_x2Face_dev, emf3_dev, is-1, ie+1, ju, sizex, hdtodx1);
__global__ void updateMagneticField_4c_dev(Real *B2_x2Face_dev, Real *emf3_dev, int is, int ie, int ju, int sizex, Real hdtodx1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Check bounds */
  if(i < is || i > ie) return;

  int ind = ju*sizex+i;

  /* Main algorithm */
  B2_x2Face_dev[ind] += hdtodx1*(emf3_dev[ind+1] - emf3_dev[ind]);

//  for (i=is-1; i<=ie+1; i++) {
//      B2_x2Face[ju][i] += hdtodx1*(emf3[ju][i+1] - emf3[ju][i]);
//    }
}

// Started as:
// updateMagneticField_9a_dev<<<nnBlocks, BLOCK_SIZE>>>(emf3_dev, pG->B1i, pG->B2i, is, ie, js, je, sizex, dtodx2, dtodx1);
__global__ void updateMagneticField_9a_dev(Real *emf3_dev, Real *B1i, Real *B2i, int is, int ie, int js, int je, int sizex, Real dtodx2, Real dtodx1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  /* Main algorithm */
  B1i[ind] -= dtodx2*(emf3_dev[ind+sizex] - emf3_dev[ind]);
  B2i[ind] += dtodx1*(emf3_dev[ind+1] - emf3_dev[ind]);

  //  for (j=js; j<=je; j++) {
  //      for (i=is; i<=ie; i++) {
  // !!!!       pG->B1i[j][i] -= dtodx2*(emf3[j+1][i  ] - emf3[j][i]);
  // !!!!       pG->B2i[j][i] += dtodx1*(emf3[j  ][i+1] - emf3[j][i]);
  //      }
  //     pG->B1i[j][ie+1] -= dtodx2*(emf3[j+1][ie+1] - emf3[j][ie+1]);
  //    }
}

// Started as:
// updateMagneticField_9b_dev<<<nBlocks_y, BLOCK_SIZE>>>(emf3_dev, pG->B1i, js, je, ie, sizex, dtodx2);
__global__ void updateMagneticField_9b_dev(Real *emf3_dev, Real *B1i, int js, int je, int ie, int sizex, Real dtodx2) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  /* Check bounds */
  if(j < js || j > je) return;

  int ind = j*sizex+ie+1;

  /* Main algorithm */
  B1i[ind] -= dtodx2*(emf3_dev[ind+sizex] - emf3_dev[ind]);

//  for (j=js; j<=je; j++) {
//      for (i=is; i<=ie; i++) {
//        pG->B1i[j][i] -= dtodx2*(emf3[j+1][i  ] - emf3[j][i]);
//        pG->B2i[j][i] += dtodx1*(emf3[j  ][i+1] - emf3[j][i]);
//      }
// !!!!    pG->B1i[j][ie+1] -= dtodx2*(emf3[j+1][ie+1] - emf3[j][ie+1]);
//    }
}

// Started as:
// updateMagneticField_9c_dev<<<nBlocks, BLOCK_SIZE>>>(emf3_dev, pG->B2i, is, ie, je, sizex, dtodx1);
__global__ void updateMagneticField_9c_dev(Real *emf3_dev, Real *B2i, int is, int ie, int je, int sizex, Real dtodx1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Check bounds */
  if(i < is || i > ie) return;

  int ind = (je+1)*sizex+i;

  /* Main algorithm */
  B2i[ind] += dtodx1*(emf3_dev[ind+1] - emf3_dev[ind]);

//  for (i=is; i<=ie; i++) {
//	  pG->B2i[je+1][i] += dtodx1*(emf3[je+1][i+1] - emf3[je+1][i]);
//  }
}



__global__ void  update_cc_mf(Gas *U, Real *B1i, Real *B2i, Real *B3i, int is, int ie, int js, int je, int sizex) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  U[ind].B1c = 0.5*(B1i[ind]+B1i[ind+1]);
  U[ind].B2c = 0.5*(B2i[ind]+B2i[(j+1)*sizex+i]);
/* Set the 3-interface magnetic field equal to the cell center field. */
  B3i[ind] = U[ind].B3c;

}

