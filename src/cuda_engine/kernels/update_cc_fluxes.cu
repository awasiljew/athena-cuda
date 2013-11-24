//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif


__global__ void update_cc_x1_Flux(Cons1D *x1Flux_dev, Gas *U, int is, int ie, int js, int je, int sizex, Real dtodx1) {

  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  /* Index for global grid array */
  int ind = j*sizex+i;

  /* Main algorithm */
  Real d = dtodx1*x1Flux_dev[ind+1].d  - dtodx1*x1Flux_dev[ind].d;
  Real M1 = dtodx1*x1Flux_dev[ind+1].Mx - dtodx1*x1Flux_dev[ind].Mx;
  Real M2 = dtodx1*x1Flux_dev[ind+1].My - dtodx1*x1Flux_dev[ind].My;
  Real M3 = dtodx1*x1Flux_dev[ind+1].Mz - dtodx1*x1Flux_dev[ind].Mz;
  Real E = dtodx1*x1Flux_dev[ind+1].E  - dtodx1*x1Flux_dev[ind].E;
  Real B2c = dtodx1*x1Flux_dev[ind+1].By - dtodx1*x1Flux_dev[ind].By;
  Real B3c = dtodx1*x1Flux_dev[ind+1].Bz - dtodx1*x1Flux_dev[ind].Bz;

  U[ind].d  -= d;
  U[ind].M1 -= M1;
  U[ind].M2 -= M2;
  U[ind].M3 -= M3;
  U[ind].E  -= E;
  U[ind].B2c -= B2c;
  U[ind].B3c -= B3c;

///////////////////////////////////////////////////////////////////////////
//  for (j=js; j<=je; j++) {
//      for (i=is; i<=ie; i++) {
//        pG->U[j][i].d  -= dtodx1*(x1Flux[j][i+1].d  - x1Flux[j][i].d );
//        pG->U[j][i].M1 -= dtodx1*(x1Flux[j][i+1].Mx - x1Flux[j][i].Mx);
//        pG->U[j][i].M2 -= dtodx1*(x1Flux[j][i+1].My - x1Flux[j][i].My);
//        pG->U[j][i].M3 -= dtodx1*(x1Flux[j][i+1].Mz - x1Flux[j][i].Mz);
//        pG->U[j][i].E  -= dtodx1*(x1Flux[j][i+1].E  - x1Flux[j][i].E );
//        pG->U[j][i].B2c -= dtodx1*(x1Flux[j][i+1].By - x1Flux[j][i].By);
//        pG->U[j][i].B3c -= dtodx1*(x1Flux[j][i+1].Bz - x1Flux[j][i].Bz);
//      }
//    }

}

//
__global__ void update_cc_x2_Flux(Cons1D *x2Flux_dev, Gas *U, int is, int ie, int js, int je, int sizex, Real dtodx2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  /* Index for global grid array */
  int ind = j*sizex+i;

#ifdef __DEVICE_EMULATION__
  //printf("update_cc_x2_Flux, %d %d %e\n", i, j, dtodx2);
#endif

  /* Main algorithm */
//  Real d = dtodx2*(x2Flux_dev[ind+sizex].d  - x2Flux_dev[ind].d);
//  Real M1 = dtodx2*(x2Flux_dev[ind+sizex].Mz - x2Flux_dev[ind].Mz);
//  Real M2 = dtodx2*(x2Flux_dev[ind+sizex].Mx - x2Flux_dev[ind].Mx);
//  Real M3 = dtodx2*(x2Flux_dev[ind+sizex].My - x2Flux_dev[ind].My);
//  Real E = dtodx2*(x2Flux_dev[ind+sizex].E  - x2Flux_dev[ind].E );
//  Real B3c = dtodx2*(x2Flux_dev[ind+sizex].By - x2Flux_dev[ind].By);
//  Real B1c = dtodx2*(x2Flux_dev[ind+sizex].Bz - x2Flux_dev[ind].Bz);
//
//  U[ind].d  -= d;
//  U[ind].M1 -= M1;
//  U[ind].M2 -= M2;
//  U[ind].M3 -= M3;
//  U[ind].E  -= E;
//  U[ind].B3c -= B3c;
//  U[ind].B1c -= B1c;

  U[ind].d  -= dtodx2*(x2Flux_dev[ind+sizex].d  - x2Flux_dev[ind].d);;
    U[ind].M1 -= dtodx2*(x2Flux_dev[ind+sizex].Mz - x2Flux_dev[ind].Mz);
    U[ind].M2 -= dtodx2*(x2Flux_dev[ind+sizex].Mx - x2Flux_dev[ind].Mx);
    U[ind].M3 -= dtodx2*(x2Flux_dev[ind+sizex].My - x2Flux_dev[ind].My);
    U[ind].E  -= dtodx2*(x2Flux_dev[ind+sizex].E  - x2Flux_dev[ind].E );
    U[ind].B3c -= dtodx2*(x2Flux_dev[ind+sizex].By - x2Flux_dev[ind].By);
    U[ind].B1c -= dtodx2*(x2Flux_dev[ind+sizex].Bz - x2Flux_dev[ind].Bz);

#ifdef __DEVICE_EMULATION__
  //printf("update_cc_x2_Flux, %e, %e, %e, %e, %e, %e, %e, %d %d %e\n",
//		  U[ind].d, U[ind].M1, U[ind].M2, U[ind].M3, U[ind].E, U[ind].B3c, U[ind].B1c, i, j, dtodx2);
#endif


///////////////////////////////////////////////////////////////////////////////
//  for (j=js; j<=je; j++) {
//      for (i=is; i<=ie; i++) {
//        pG->U[j][i].d  -= dtodx2*(x2Flux[j+1][i].d  - x2Flux[j][i].d );
//        pG->U[j][i].M1 -= dtodx2*(x2Flux[j+1][i].Mz - x2Flux[j][i].Mz);
//        pG->U[j][i].M2 -= dtodx2*(x2Flux[j+1][i].Mx - x2Flux[j][i].Mx);
//        pG->U[j][i].M3 -= dtodx2*(x2Flux[j+1][i].My - x2Flux[j][i].My);
//        pG->U[j][i].E  -= dtodx2*(x2Flux[j+1][i].E  - x2Flux[j][i].E );
//        pG->U[j][i].B3c -= dtodx2*(x2Flux[j+1][i].By - x2Flux[j][i].By);
//        pG->U[j][i].B1c -= dtodx2*(x2Flux[j+1][i].Bz - x2Flux[j][i].Bz);
//      }
//    }

}
