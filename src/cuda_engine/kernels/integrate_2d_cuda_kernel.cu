//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include "../defs.h"
//#include "../athena.h"
//#include "../globals.h"
//#include "../prototypes.h"
//
//#include <cuda.h>
//#include <cuda_runtime.h>
//
//
///************** Device variables ***************/
///* The L/R states of conserved variables and fluxes at each cell face */
//static __device__ Cons1D *Ul_x1Face_dev=NULL, *Ur_x1Face_dev=NULL;
//static __device__ Cons1D *Ul_x2Face_dev=NULL, *Ur_x2Face_dev=NULL;
//static __device__ Cons1D *x1Flux_dev=NULL, *x2Flux_dev=NULL;
//
///* The interface magnetic fields and emfs */
//static __device__ Real *B1_x1Face_dev=NULL, *B2_x2Face_dev=NULL;
//static __device__ Real *emf3_dev=NULL, *emf3_cc_dev=NULL;
//
///* 1D scratch vectors used by lr_states and flux functions */
//static __device__ Real *Bxc_dev=NULL, *Bxi_dev=NULL;
//static __device__ Prim1D *W_dev=NULL, *Wl_dev=NULL, *Wr_dev=NULL;
//static __device__ Cons1D *U1d_dev=NULL, *Ul_dev=NULL, *Ur_dev=NULL;
//
///* density at t^{n+1/2} needed by both MHD and to make gravity 2nd order */
//static __device__ Real *dhalf_dev = NULL;


/**
 * j=0..NY-1
 * i=0..NX-1
 */
__global__ void stepA(Grid_gpu *pG, int NX, int j) {
  /* Cell index */
  int i = blockDim.x + blockIdx.x + threadIdx.x;

  /*--- Step 1a ------------------------------------------------------------------
   * Load 1D vector of conserved variables;
   * U1d = (d, M1, M2, M3, E, B2c, B3c, s[n])
   */
  U1d_dev[i].d = pG->U[j*NX+i].d;
  U1d_dev[i].Mx = pG->U[j*NX+i].M1;
  U1d_dev[i].My = pG->U[j*NX+i].M2;
  U1d_dev[i].Mz = pG->U[j*NX+i].M3;
  U1d_dev[i].E = pG->U[j*NX+i].E;
  U1d_dev[i].By = pG->U[j*NX+i].B2c;
  U1d_dev[i].Bz = pG->U[j*NX+i].B3c;
  Bxc_dev[i] = pG->U[j*NX+i].B1c;
  Bxi_dev[i] = pG->B1i[j*NX+i];
  B1_x1Face_dev[j*NX+i] = pG->B1i[j*NX+i];

  /*--- Step 1b ------------------------------------------------------------------
   * Compute L and R states at X1-interfaces.
   */
  //TODO:
  //Cons1D_to_Prim1D(&U1d_dev[i], &W_dev[i], &Bxc_dev[i]);
}

/**
 * Second milestone is a normal function - stores left right states in W_l and W_r need to compute fluxes
 */
void stepB() {
  //TODO: its not inside loop!!!
  //lr_states(W_dev, Bxc_dev, pG->dt, dtodx1, is-1, ie+1, Wl_dev, Wr_dev);
}

/**
 * Compute fluxes :D
 * j=0..NY-1
 * i=1..NX-1
 */
__global__ void stepC(Grid_gpu *pG, int j, int NX, Real hdt) {
  /* Add "MHD source terms" for 0.5*dt */
  Real MHD_src;
  /* Cell index */
  int i = blockDim.x + blockIdx.x + threadIdx.x+1; //Do not begin from first cell

  MHD_src = (pG->U[j*NX+i-1].M2/pG->U[j*NX+i-1].d)*
      (pG->B1i[j*NX+i] - pG->B1i[j*NX+i-1])/pG->dx1;
  Wl_dev[i].By += hdt*MHD_src;

  MHD_src = (pG->U[j*NX+i].M2/pG->U[j*NX+i].d)*
      (pG->B1i[j*NX+i+1] - pG->B1i[j*NX+i])/pG->dx1;
  Wr_dev[i].By += hdt*MHD_src;

  /*--- Step 1e ------------------------------------------------------------------
   * Compute 1D fluxes in x1-direction, storing into 2D array
   */

  //TODO:
  //Prim1D_to_Cons1D(&Ul_x1Face_dev[j*NX+i],&Wl_dev[i],&Bxi_dev[i]);
  //Prim1D_to_Cons1D(&Ur_x1Face_dev[j*NX+i],&Wr_dev[i],&Bxi_dev[i]);

  //flux_roe(Ul_x1Face[j*NX+i],Ur_x1Face[j*NX+i],Wl[i],Wr[i],
  //    B1_x1Face[j*NX+i],&x1Flux[j*NX+i]);
}


//void integrate(Grid_gpu *pG) {
//  //integrate
//  Real dtodx1 = pG->dt/pG->dx1, dtodx2 = pG->dt/pG->dx2;
//  Real hdtodx1 = 0.5*dtodx1, hdtodx2 = 0.5*dtodx2;
//  int is = pG->is, ie = pG->ie;
//  int js = pG->js, je = pG->je;
//  //int ks = pG->ks;
//  int i,il,iu;
//  int j,jl,ju;
//  Real MHD_src,dbx,dby,B1,B2,B3,V3;
//  Real d, M1, M2, B1c, B2c;
//  Real hdt = 0.5*pG->dt;
//  Real x1,x2,x3,phicl,phicr,phifc,phil,phir,phic;
//
//  //cells in x direction == nx1 + 2*nghost;
//
//  // My Definitions:
//  int NX = nx1+2*nghost;
//  int NY = nx2+2*nghost;
//  int SIZE_NXNY = NX*NY;
//
//  il = is - 2; //0
//  iu = ie + 2; //NX-1
//
//  jl = js - 2; //0
//  ju = je + 2; //NY-1
//
///////////////////////////////////////////////////////////////////////////////////
//// FIRST PATCH                                                                 //
//// Kernel execution - every row as a single kernel                             //
//// int threadsPerBlock = 256;                                                  //
//// int blocksPerGrid = (SIZE + threadsPerBlock - 1)/threadsPerBlock;           //
//// invoke<<<blocksPerGrid, threadsPerBlock>>>(...)                             //
//// Where SIZE is NX and invoke in loop NY times                                //
///////////////////////////////////////////////////////////////////////////////////
//
///*--- Step 1a ------------------------------------------------------------------
// * Load 1D vector of conserved variables;
// * U1d = (d, M1, M2, M3, E, B2c, B3c, s[n])
// */
//  //printf("STEP 1\n");
//
//  //states_and_fluxes1D_cu(pG);
//  for (j=jl; j<=ju; j++) { //rows
//
//    //Columns
//    //Get 1 whole row of memory to 1D vector
//    for(i=is-nghost; i<=ie+nghost; i++) {
//      U1d_dev[i].d = pG->U[j*NX+i].d;
//      U1d_dev[i].Mx = pG->U[j*NX+i].M1;
//      U1d_dev[i].My = pG->U[j*NX+i].M2;
//      U1d_dev[i].Mz = pG->U[j*NX+i].M3;
//      U1d_dev[i].E = pG->U[j*NX+i].E;
//      U1d_dev[i].By = pG->U[j*NX+i].By;
//      U1d_dev[i].Bz = pG->U[j*NX+i].Bz;
//      Bxc_dev[i] = pG->U[j*NX+i].B1c;
//      Bxi_dev[i] = pG->B1i[j*NX+i];
//      B1_x1Face_dev[j*NX+i] = pG->B1i[j*NX+i];
//    }
//
///*--- Step 1b ------------------------------------------------------------------
// * Compute L and R states at X1-interfaces.
// */
//
//    for(i=is-nghost; i<=ie+nghost; i++) {
//      //TODO:
//      //Cons1D_to_Prim1D(&U1d_dev[i], &W_dev[i], &Bxc_dev[i]);
//    }
//
//    //TODO:
//    //lr_states(W_dev, Bxc_dev, pG->dt, dtodx1, is-1, ie+1, Wl_dev, Wr_dev);
//
//    /* Add "MHD source terms" for 0.5*dt */
//
//    for (i=is-1; i<=iu; i++) {
//      MHD_src = (pG->U[j*NX+i-1].M2/pG->U[j*NX+i-1].d)*
//               (pG->B1i[j*NX+i] - pG->B1i[j*NX+i-1])/pG->dx1;
//      Wl_dev[i].By += hdt*MHD_src;
//
//      MHD_src = (pG->U[j*NX+i].M2/pG->U[j*NX+i].d)*
//               (pG->B1i[j*NX+i+1] - pG->B1i[j*NX+i])/pG->dx1;
//      Wr_dev[i].By += hdt*MHD_src;
//    }
//
//
//
///*--- Step 1e ------------------------------------------------------------------
// * Compute 1D fluxes in x1-direction, storing into 2D array
// */
//
//    for (i=is-1; i<=iu; i++) {
//      //TODO:
//      //Prim1D_to_Cons1D(&Ul_x1Face_dev[j*NX+i],&Wl_dev[i],&Bxi_dev[i]);
//      //Prim1D_to_Cons1D(&Ur_x1Face_dev[j*NX+i],&Wr_dev[i],&Bxi_dev[i]);
//
//      //flux_roe(Ul_x1Face[j*NX+i],Ur_x1Face[j*NX+i],Wl[i],Wr[i],
//      //    B1_x1Face[j*NX+i],&x1Flux[j*NX+i]);
//    }
//  }
//
////////////////////////////////////////////////////////////////////////////////////////
////                     END OF FIRST PATCH                                           //
////////////////////////////////////////////////////////////////////////////////////////
//
//
//
///*--- Step 2a ------------------------------------------------------------------
// * Load 1D vector of conserved variables;
// * U1d = (d, M2, M3, M1, E, B3c, B1c, s[n])
// */
//
//  for (i=il; i<=iu; i++) {
//    for (j=js-nghost; j<=je+nghost; j++) {
//      U1d[j].d  = pG->U[j][i].d;
//      U1d[j].Mx = pG->U[j][i].M2;
//      U1d[j].My = pG->U[j][i].M3;
//      U1d[j].Mz = pG->U[j][i].M1;
//      U1d[j].E  = pG->U[j][i].E;
//      U1d[j].By = pG->U[j][i].B3c;
//      U1d[j].Bz = pG->U[j][i].B1c;
//      Bxc[j] = pG->U[j][i].B2c;
//      Bxi[j] = pG->B2i[j][i];
//      B2_x2Face[j][i] = pG->B2i[j][i];
//    }
//
///*--- Step 2b ------------------------------------------------------------------
// * Compute L and R states at X2-interfaces.
// */
//
//    for (j=js-nghost; j<=je+nghost; j++) {
//      Cons1D_to_Prim1D(&U1d[j],&W[j],&Bxc[j]);
//    }
//    lr_states(W,Bxc,pG->dt,dtodx2,js-1,je+1,Wl,Wr);
//
///* Add "MHD source terms" for 0.5*dt */
//    for (j=js-1; j<=ju; j++) {
//      MHD_src = (pG->U[j-1][i].M1/pG->U[j-1][i].d)*
//        (pG->B2i[j][i] - pG->B2i[j-1][i])/pG->dx2;
//      Wl[j].Bz += hdt*MHD_src;
//
//      MHD_src = (pG->U[j][i].M1/pG->U[j][i].d)*
//        (pG->B2i[j+1][i] - pG->B2i[j][i])/pG->dx2;
//      Wr[j].Bz += hdt*MHD_src;
//    }
//
//
///*--- Step 2e ------------------------------------------------------------------
// * Compute 1D fluxes in x2-direction, storing into 2D array
// */
//
//    for (j=js-1; j<=ju; j++) {
//      Prim1D_to_Cons1D(&Ul_x2Face[j][i],&Wl[j],&Bxi[j]);
//      Prim1D_to_Cons1D(&Ur_x2Face[j][i],&Wr[j],&Bxi[j]);
//
//      flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[j],Wr[j],
//                 B2_x2Face[j][i],&x2Flux[j][i]);
//    }
//  }
//
///*--- Step 3 ------------------------------------------------------------------
// * Calculate the cell centered value of emf_3 at t^{n}
// */
//
//  for (j=jl; j<=ju; j++) {
//    for (i=il; i<=iu; i++) {
//      emf3_cc[j][i] =
//        (pG->U[j][i].B1c*pG->U[j][i].M2 -
//         pG->U[j][i].B2c*pG->U[j][i].M1 )/pG->U[j][i].d;
//    }
//  }
//
///*--- Step 4 ------------------------------------------------------------------
// * Integrate emf3 to the grid cell corners and then update the
// * interface magnetic fields using CT for a half time step.
// */
//
//  integrate_emf3_corner(pG);
//
//  for (j=js-1; j<=je+1; j++) {
//    for (i=is-1; i<=ie+1; i++) {
//      B1_x1Face[j][i] -= hdtodx2*(emf3[j+1][i  ] - emf3[j][i]);
//      B2_x2Face[j][i] += hdtodx1*(emf3[j  ][i+1] - emf3[j][i]);
//    }
//    B1_x1Face[j][iu] -= hdtodx2*(emf3[j+1][iu] - emf3[j][iu]);
//  }
//  for (i=is-1; i<=ie+1; i++) {
//    B2_x2Face[ju][i] += hdtodx1*(emf3[ju][i+1] - emf3[ju][i]);
//  }
//
///*--- Step 5a ------------------------------------------------------------------
// * Correct the L/R states at x1-interfaces using transverse flux-gradients in
// * the x2-direction for 0.5*dt using x2-fluxes computed in Step 2e.
// * Since the fluxes come from an x2-sweep, (x,y,z) on RHS -> (z,x,y) on LHS */
//
//  for (j=js-1; j<=je+1; j++) {
//    for (i=is-1; i<=iu; i++) {
//      Ul_x1Face[j][i].d  -= hdtodx2*(x2Flux[j+1][i-1].d  - x2Flux[j][i-1].d );
//      Ul_x1Face[j][i].Mx -= hdtodx2*(x2Flux[j+1][i-1].Mz - x2Flux[j][i-1].Mz);
//      Ul_x1Face[j][i].My -= hdtodx2*(x2Flux[j+1][i-1].Mx - x2Flux[j][i-1].Mx);
//      Ul_x1Face[j][i].Mz -= hdtodx2*(x2Flux[j+1][i-1].My - x2Flux[j][i-1].My);
//      Ul_x1Face[j][i].E  -= hdtodx2*(x2Flux[j+1][i-1].E  - x2Flux[j][i-1].E );
//      Ul_x1Face[j][i].Bz -= hdtodx2*(x2Flux[j+1][i-1].By - x2Flux[j][i-1].By);
//
//      Ur_x1Face[j][i].d  -= hdtodx2*(x2Flux[j+1][i  ].d  - x2Flux[j][i  ].d );
//      Ur_x1Face[j][i].Mx -= hdtodx2*(x2Flux[j+1][i  ].Mz - x2Flux[j][i  ].Mz);
//      Ur_x1Face[j][i].My -= hdtodx2*(x2Flux[j+1][i  ].Mx - x2Flux[j][i  ].Mx);
//      Ur_x1Face[j][i].Mz -= hdtodx2*(x2Flux[j+1][i  ].My - x2Flux[j][i  ].My);
//      Ur_x1Face[j][i].E  -= hdtodx2*(x2Flux[j+1][i  ].E  - x2Flux[j][i  ].E );
//      Ur_x1Face[j][i].Bz -= hdtodx2*(x2Flux[j+1][i  ].By - x2Flux[j][i  ].By);
//    }
//  }
//
///*--- Step 5b ------------------------------------------------------------------
// * Add the "MHD source terms" to the x2 (conservative) flux gradient.
// */
//
//  for (j=js-1; j<=je+1; j++) {
//    for (i=is-1; i<=iu; i++) {
//      dbx = pG->B1i[j][i] - pG->B1i[j][i-1];
//      B1 = pG->U[j][i-1].B1c;
//      B2 = pG->U[j][i-1].B2c;
//      B3 = pG->U[j][i-1].B3c;
//      V3 = pG->U[j][i-1].M3/pG->U[j][i-1].d;
//
//      Ul_x1Face[j][i].Mx += hdtodx1*B1*dbx;
//      Ul_x1Face[j][i].My += hdtodx1*B2*dbx;
//      Ul_x1Face[j][i].Mz += hdtodx1*B3*dbx;
//      Ul_x1Face[j][i].Bz += hdtodx1*V3*dbx;
//      Ul_x1Face[j][i].E  += hdtodx1*B3*V3*dbx;
//
//      dbx = pG->B1i[j][i+1] - pG->B1i[j][i];
//      B1 = pG->U[j][i].B1c;
//      B2 = pG->U[j][i].B2c;
//      B3 = pG->U[j][i].B3c;
//      V3 = pG->U[j][i].M3/pG->U[j][i].d;
//
//      Ur_x1Face[j][i].Mx += hdtodx1*B1*dbx;
//      Ur_x1Face[j][i].My += hdtodx1*B2*dbx;
//      Ur_x1Face[j][i].Mz += hdtodx1*B3*dbx;
//      Ur_x1Face[j][i].Bz += hdtodx1*V3*dbx;
//      Ur_x1Face[j][i].E  += hdtodx1*B3*V3*dbx;
//    }
//  }
//
//
///*--- Step 6a ------------------------------------------------------------------
// * Correct the L/R states at x2-interfaces using transverse flux-gradients in
// * the x1-direction for 0.5*dt using x1-fluxes computed in Step 1e.
// * Since the fluxes come from an x1-sweep, (x,y,z) on RHS -> (y,z,x) on LHS */
//
//  for (j=js-1; j<=ju; j++) {
//    for (i=is-1; i<=ie+1; i++) {
//      Ul_x2Face[j][i].d  -= hdtodx1*(x1Flux[j-1][i+1].d  - x1Flux[j-1][i].d );
//      Ul_x2Face[j][i].Mx -= hdtodx1*(x1Flux[j-1][i+1].My - x1Flux[j-1][i].My);
//      Ul_x2Face[j][i].My -= hdtodx1*(x1Flux[j-1][i+1].Mz - x1Flux[j-1][i].Mz);
//      Ul_x2Face[j][i].Mz -= hdtodx1*(x1Flux[j-1][i+1].Mx - x1Flux[j-1][i].Mx);
//      Ul_x2Face[j][i].E  -= hdtodx1*(x1Flux[j-1][i+1].E  - x1Flux[j-1][i].E );
//      Ul_x2Face[j][i].By -= hdtodx1*(x1Flux[j-1][i+1].Bz - x1Flux[j-1][i].Bz);
//
//      Ur_x2Face[j][i].d  -= hdtodx1*(x1Flux[j  ][i+1].d  - x1Flux[j  ][i].d );
//      Ur_x2Face[j][i].Mx -= hdtodx1*(x1Flux[j  ][i+1].My - x1Flux[j  ][i].My);
//      Ur_x2Face[j][i].My -= hdtodx1*(x1Flux[j  ][i+1].Mz - x1Flux[j  ][i].Mz);
//      Ur_x2Face[j][i].Mz -= hdtodx1*(x1Flux[j  ][i+1].Mx - x1Flux[j  ][i].Mx);
//      Ur_x2Face[j][i].E  -= hdtodx1*(x1Flux[j  ][i+1].E  - x1Flux[j  ][i].E );
//      Ur_x2Face[j][i].By -= hdtodx1*(x1Flux[j][i+1].Bz - x1Flux[j][i].Bz);
//    }
//  }
//
///*--- Step 6b ------------------------------------------------------------------
// * Add the "MHD source terms" to the x1 (conservative) flux gradient.
// */
//
//  for (j=js-1; j<=ju; j++) {
//    for (i=is-1; i<=ie+1; i++) {
//      dby = pG->B2i[j][i] - pG->B2i[j-1][i];
//      B1 = pG->U[j-1][i].B1c;
//      B2 = pG->U[j-1][i].B2c;
//      B3 = pG->U[j-1][i].B3c;
//      V3 = pG->U[j-1][i].M3/pG->U[j-1][i].d;
//
//      Ul_x2Face[j][i].Mz += hdtodx2*B1*dby;
//      Ul_x2Face[j][i].Mx += hdtodx2*B2*dby;
//      Ul_x2Face[j][i].My += hdtodx2*B3*dby;
//      Ul_x2Face[j][i].By += hdtodx2*V3*dby;
//      Ul_x2Face[j][i].E  += hdtodx2*B3*V3*dby;
//
//      dby = pG->B2i[j+1][i] - pG->B2i[j][i];
//      B1 = pG->U[j][i].B1c;
//      B2 = pG->U[j][i].B2c;
//      B3 = pG->U[j][i].B3c;
//      V3 = pG->U[j][i].M3/pG->U[j][i].d;
//
//      Ur_x2Face[j][i].Mz += hdtodx2*B1*dby;
//      Ur_x2Face[j][i].Mx += hdtodx2*B2*dby;
//      Ur_x2Face[j][i].My += hdtodx2*B3*dby;
//      Ur_x2Face[j][i].By += hdtodx2*V3*dby;
//      Ur_x2Face[j][i].E  += hdtodx2*B3*V3*dby;
//    }
//  }
//
//
///*--- Step 7 ------------------------------------------------------------------
// * Calculate the cell centered value of emf_3 at t^{n+1/2}, needed by CT
// * algorithm to integrate emf to corner in step 10
// */
//
//  if (dhalf != NULL){
//    for (j=js-1; j<=je+1; j++) {
//      for (i=is-1; i<=ie+1; i++) {
//        dhalf[j][i] = pG->U[j][i].d
//          - hdtodx1*(x1Flux[j  ][i+1].d - x1Flux[j][i].d)
//          - hdtodx2*(x2Flux[j+1][i  ].d - x2Flux[j][i].d);
//      }
//    }
//  }
//
//  for (j=js-1; j<=je+1; j++) {
//    for (i=is-1; i<=ie+1; i++) {
//      cc_pos(pG,i,j,&x1,&x2/*,&x3*/);
//
//      d  = dhalf[j][i];
//
//      M1 = pG->U[j][i].M1
//        - hdtodx1*(x1Flux[j][i+1].Mx - x1Flux[j][i].Mx)
//        - hdtodx2*(x2Flux[j+1][i].Mz - x2Flux[j][i].Mz);
//
//      M2 = pG->U[j][i].M2
//        - hdtodx1*(x1Flux[j][i+1].My - x1Flux[j][i].My)
//        - hdtodx2*(x2Flux[j+1][i].Mx - x2Flux[j][i].Mx);
//
//      B1c = 0.5*(B1_x1Face[j][i] + B1_x1Face[j][i+1]);
//      B2c = 0.5*(B2_x2Face[j][i] + B2_x2Face[j+1][i]);
//
//      emf3_cc[j][i] = (B1c*M2 - B2c*M1)/d;
//    }
//  }
//
//
///*--- Step 8b ------------------------------------------------------------------
// * Compute x1-fluxes from corrected L/R states.
// */
//
//  for (j=js-1; j<=je+1; j++) {
//    for (i=is; i<=ie+1; i++) {
//
//      Cons1D_to_Prim1D(&Ul_x1Face[j][i],&Wl[i],&B1_x1Face[j][i]);
//      Cons1D_to_Prim1D(&Ur_x1Face[j][i],&Wr[i],&B1_x1Face[j][i]);
//
//      flux_roe(Ul_x1Face[j][i],Ur_x1Face[j][i],Wl[i],Wr[i],
//                 B1_x1Face[j][i],&x1Flux[j][i]);
//    }
//  }
//
///*--- Step 8c ------------------------------------------------------------------
// * Compute x2-fluxes from corrected L/R states.
// */
//
//  for (j=js; j<=je+1; j++) {
//    for (i=is-1; i<=ie+1; i++) {
//
//      Cons1D_to_Prim1D(&Ul_x2Face[j][i],&Wl[i],&B2_x2Face[j][i]);
//      Cons1D_to_Prim1D(&Ur_x2Face[j][i],&Wr[i],&B2_x2Face[j][i]);
//
//      flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[i],Wr[i],
//                 B2_x2Face[j][i],&x2Flux[j][i]);
//    }
//  }
//
///*--- Step 9 -------------------------------------------------------------------
// * Integrate emf3^{n+1/2} to the grid cell corners and then update the
// * interface magnetic fields using CT for a full time step.
// */
//
//  integrate_emf3_corner(pG);
//
//  for (j=js; j<=je; j++) {
//    for (i=is; i<=ie; i++) {
//      pG->B1i[j][i] -= dtodx2*(emf3[j+1][i  ] - emf3[j][i]);
//      pG->B2i[j][i] += dtodx1*(emf3[j  ][i+1] - emf3[j][i]);
//    }
//    pG->B1i[j][ie+1] -= dtodx2*(emf3[j+1][ie+1] - emf3[j][ie+1]);
//  }
//  for (i=is; i<=ie; i++) {
//    pG->B2i[je+1][i] += dtodx1*(emf3[je+1][i+1] - emf3[je+1][i]);
//  }
//
//
///*--- Step 11a -----------------------------------------------------------------
// * Update cell-centered variables in pG using x1-fluxes
// */
//
//  for (j=js; j<=je; j++) {
//    for (i=is; i<=ie; i++) {
//      pG->U[j][i].d  -= dtodx1*(x1Flux[j][i+1].d  - x1Flux[j][i].d );
//      pG->U[j][i].M1 -= dtodx1*(x1Flux[j][i+1].Mx - x1Flux[j][i].Mx);
//      pG->U[j][i].M2 -= dtodx1*(x1Flux[j][i+1].My - x1Flux[j][i].My);
//      pG->U[j][i].M3 -= dtodx1*(x1Flux[j][i+1].Mz - x1Flux[j][i].Mz);
//      pG->U[j][i].E  -= dtodx1*(x1Flux[j][i+1].E  - x1Flux[j][i].E );
//      pG->U[j][i].B2c -= dtodx1*(x1Flux[j][i+1].By - x1Flux[j][i].By);
//      pG->U[j][i].B3c -= dtodx1*(x1Flux[j][i+1].Bz - x1Flux[j][i].Bz);
//    }
//  }
//
///*--- Step 11b -----------------------------------------------------------------
// * Update cell-centered variables in pG using x2-fluxes
// */
//
//  for (j=js; j<=je; j++) {
//    for (i=is; i<=ie; i++) {
//      pG->U[j][i].d  -= dtodx2*(x2Flux[j+1][i].d  - x2Flux[j][i].d );
//      pG->U[j][i].M1 -= dtodx2*(x2Flux[j+1][i].Mz - x2Flux[j][i].Mz);
//      pG->U[j][i].M2 -= dtodx2*(x2Flux[j+1][i].Mx - x2Flux[j][i].Mx);
//      pG->U[j][i].M3 -= dtodx2*(x2Flux[j+1][i].My - x2Flux[j][i].My);
//      pG->U[j][i].E  -= dtodx2*(x2Flux[j+1][i].E  - x2Flux[j][i].E );
//      pG->U[j][i].B3c -= dtodx2*(x2Flux[j+1][i].By - x2Flux[j][i].By);
//      pG->U[j][i].B1c -= dtodx2*(x2Flux[j+1][i].Bz - x2Flux[j][i].Bz);
//    }
//  }
//
///*--- Step 13 ------------------------------------------------------------------
// * LAST STEP!
// * Set cell centered magnetic fields to average of updated face centered fields.
// */
//
//  for (j=js; j<=je; j++) {
//    for (i=is; i<=ie; i++) {
//
//      pG->U[j][i].B1c =0.5*(pG->B1i[j][i]+pG->B1i[j][i+1]);
//      pG->U[j][i].B2c =0.5*(pG->B2i[j][i]+pG->B2i[j+1][i]);
///* Set the 3-interface magnetic field equal to the cell center field. */
//      pG->B3i[j][i] = pG->U[j][i].B3c;
//    }
//  }
//
//  return;
//}
