/* DONE - OK */

/**
 * Modified by awasiljew -> awasiljew@gmail.com
 * 22.11.2009
 * We assuming MHD, adiabatic equation of state, no H_CORRECTION, NSCALARS =0, NO_SELF_GRAVITY, SELF_GRAVITY_NONE,
 * GET_FLUXES=flux_roe - Roe Flux, StaticGravPot = NULL
 */


#include "copyright.h"
/*==============================================================================
 * FILE: integrate_2d.c
 *
 * PURPOSE: Updates the input Grid structure pointed to by *pG by one 
 *   timestep using directionally unsplit CTU method of Colella (1990).  The
 *   variables updated are:
 *      U.[d,M1,M2,M3,E,B1c,B2c,B3c,s] -- where U is of type Gas
 *      B1i, B2i -- interface magnetic field
 *   Also adds gravitational source terms, self-gravity, and H-correction
 *   of Sanders et al.
 *
 * REFERENCES:
 *   P. Colella, "Multidimensional upwind methods for hyperbolic conservation
 *   laws", JCP, 87, 171 (1990)
 *
 *   T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
 *   constrained transport", JCP, 205, 509 (2005)
 *
 *   R. Sanders, E. Morano, & M.-C. Druguet, "Multidimensinal dissipation for
 *   upwind schemes: stability and applications to gas dynamics", JCP, 145, 511
 *   (1998)
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   integrate_2d()
 *   integrate_init_2d()
 *   integrate_destruct_2d()
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "debug_tools_cuda.h"

/* The L/R states of conserved variables and fluxes at each cell face */
/*static*/ Cons1D **Ul_x1Face=NULL, **Ur_x1Face=NULL;
/* static */ Cons1D **Ul_x2Face=NULL, **Ur_x2Face=NULL;
/*static*/ Cons1D **x1Flux=NULL, **x2Flux=NULL;

/* The interface magnetic fields and emfs */
/* static */ Real **B1_x1Face=NULL, **B2_x2Face=NULL;
/* static */ Real **emf3=NULL, **emf3_cc=NULL;

/* 1D scratch vectors used by lr_states and flux functions */
/*static*/ Real **Bxc=NULL, **Bxi=NULL;
/*static*/ Prim1D **W=NULL, **Wl=NULL, **Wr=NULL;
/*static*/ Cons1D **U1d=NULL, *Ul=NULL, *Ur=NULL;

/* density at t^{n+1/2} needed by both MHD and to make gravity 2nd order */
static Real **dhalf = NULL;

/* variables needed for H-correction of Sanders et al (1998) */
extern Real etah;


/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES: 
 *   integrate_emf3_corner() - the upwind CT method in Gardiner & Stone (2005) 
 *============================================================================*/

static void integrate_emf3_corner(Grid *pG);

/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/
/* integrate_2d:  CTU integrator in 2D  */

double printTime(struct timeval t1,struct timeval t2, const char* process) {
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    printf("%s : %e \n", process, time);
    return time;
}

double printTimeSpeedup(struct timeval t1,struct timeval t2, struct timeval t1_gpu,struct timeval t2_gpu, const char* process) {
    double time1 = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    double time2 = (1000000.0*(t2_gpu.tv_sec-t1_gpu.tv_sec) + t2_gpu.tv_usec-t1_gpu.tv_usec)/1000000.0;
    printf("%s : %f x time faster\n", process, time1/time2);
    return time1/time2;
}

void integrate_2d(Grid *pG)
{
  Real dtodx1 = pG->dt/pG->dx1, dtodx2 = pG->dt/pG->dx2;
  Real hdtodx1 = 0.5*dtodx1, hdtodx2 = 0.5*dtodx2;
  int is = pG->is, ie = pG->ie;
  int js = pG->js, je = pG->je;
  int i,il,iu;
  int j,jl,ju;
  Real MHD_src,dbx,dby,B1,B2,B3,V3;
  Real d, M1, M2, B1c, B2c;
  Real hdt = 0.5*pG->dt;
  Real x1,x2,x3,phicl,phicr,phifc,phil,phir,phic;

  int sizex = pG->Nx1+2*nghost;
  int sizey = pG->Nx2+2*nghost;

  il = is - 2;
  iu = ie + 2;

  jl = js - 2;
  ju = je + 2;

/*--- Step 1a ------------------------------------------------------------------
 * Load 1D vector of conserved variables;
 * U1d = (d, M1, M2, M3, E, B2c, B3c, s[n])
 */


  for (j=jl; j<=ju; j++) {

	for (i=is-nghost; i<=ie+nghost; i++) {
      U1d[j][i].d  = pG->U[j][i].d;
      U1d[j][i].Mx = pG->U[j][i].M1;
      U1d[j][i].My = pG->U[j][i].M2;
      U1d[j][i].Mz = pG->U[j][i].M3;
      U1d[j][i].E  = pG->U[j][i].E;
      U1d[j][i].By = pG->U[j][i].B2c;
      U1d[j][i].Bz = pG->U[j][i].B3c;
      Bxc[j][i] = pG->U[j][i].B1c;
      B1_x1Face[j][i] = pG->B1i[j][i];
    }


/*--- Step 1b ------------------------------------------------------------------
 * Compute L and R states at X1-interfaces.
 */

    for (i=is-nghost; i<=ie+nghost; i++) {
      Cons1D_to_Prim1D(&U1d[j][i],&W[j][i],&Bxc[j][i]);
    }

    lr_states(W[j],Bxc[j],pG->dt,dtodx1,is-1,ie+1,Wl[j],Wr[j]);


/* Add "MHD source terms" for 0.5*dt */

    for (i=is-1; i<=iu; i++) {
      MHD_src = (pG->U[j][i-1].M2/pG->U[j][i-1].d)*
               (pG->B1i[j][i] - pG->B1i[j][i-1])/pG->dx1;
      Wl[j][i].By += hdt*MHD_src;

      MHD_src = (pG->U[j][i].M2/pG->U[j][i].d)*
               (pG->B1i[j][i+1] - pG->B1i[j][i])/pG->dx1;
      Wr[j][i].By += hdt*MHD_src;
    }

  }

  /*--- Step 1e ------------------------------------------------------------------
   * Compute 1D fluxes in x1-direction, storing into 2D array
   */

  for (j=jl; j<=ju; j++) {
  for (i=is-1; i<=iu; i++) {

	  Prim1D_to_Cons1D(&Ul_x1Face[j][i],&Wl[j][i],&(pG->B1i[j][i]));
      Prim1D_to_Cons1D(&Ur_x1Face[j][i],&Wr[j][i],&(pG->B1i[j][i]));

        flux_roe(Ul_x1Face[j][i],Ur_x1Face[j][i],Wl[j][i],Wr[j][i],
        		B1_x1Face[j][i],&x1Flux[j][i]);
  }}


/*--- Step 2a ------------------------------------------------------------------
 * Load 1D vector of conserved variables;
 * U1d = (d, M2, M3, M1, E, B3c, B1c, s[n])
 */


  for (i=il; i<=iu; i++) {

    for (j=js-nghost; j<=je+nghost; j++) {
      U1d[j][i].d  = pG->U[j][i].d;
      U1d[j][i].Mx = pG->U[j][i].M2;
      U1d[j][i].My = pG->U[j][i].M3;
      U1d[j][i].Mz = pG->U[j][i].M1;
      U1d[j][i].E  = pG->U[j][i].E;
      U1d[j][i].By = pG->U[j][i].B3c;
      U1d[j][i].Bz = pG->U[j][i].B1c;
      Bxc[i][j] = pG->U[j][i].B2c; //Rows -> columns : Bxc sizex x sizey
      Bxi[j][i] = pG->B2i[j][i];
      B2_x2Face[j][i] = pG->B2i[j][i];
    }

/*--- Step 2b ------------------------------------------------------------------
 * Compute L and R states at X2-interfaces.
 */


    for (j=js-nghost; j<=je+nghost; j++) {
      Cons1D_to_Prim1D(&U1d[j][i],&W[i][j],&Bxc[i][j]); //W and Bxc rows -> columns : sizex x sizey
    }
    lr_states(W[i],Bxc[i],pG->dt,dtodx2,js-1,je+1,Wl[i],Wr[i]); // over j indexes and fixed i, but W, Bxc rows->columns: sizex x sizey, and so its Wl and Wr


/* Add "MHD source terms" for 0.5*dt */
    for (j=js-1; j<=ju; j++) {
      MHD_src = (pG->U[j-1][i].M1/pG->U[j-1][i].d)*
        (pG->B2i[j][i] - pG->B2i[j-1][i])/pG->dx2;
      Wl[i][j].Bz += hdt*MHD_src;

      MHD_src = (pG->U[j][i].M1/pG->U[j][i].d)*
        (pG->B2i[j+1][i] - pG->B2i[j][i])/pG->dx2;
      Wr[i][j].Bz += hdt*MHD_src;
    }

  }


  /*--- Step 2e ------------------------------------------------------------------
   * Compute 1D fluxes in x2-direction, storing into 2D array
   */

  for (i=il; i<=iu; i++) {
  for (j=js-1; j<=ju; j++) {
        Prim1D_to_Cons1D(&Ul_x2Face[j][i],&Wl[i][j],&(pG->B2i[j][i]));
        Prim1D_to_Cons1D(&Ur_x2Face[j][i],&Wr[i][j],&(pG->B2i[j][i]));

        flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[i][j],Wr[i][j],
                   B2_x2Face[j][i],&x2Flux[j][i]);
      }
  }


  /*--- Step 3 ------------------------------------------------------------------
   * Calculate the cell centered value of emf_3 at t^{n}
   */


    for (j=jl; j<=ju; j++) {
      for (i=il; i<=iu; i++) {
        emf3_cc[j][i] =
  	(pG->U[j][i].B1c*pG->U[j][i].M2 -
  	 pG->U[j][i].B2c*pG->U[j][i].M1 )/pG->U[j][i].d;
      }
    }


  /*--- Step 4 ------------------------------------------------------------------
   * Integrate emf3 to the grid cell corners and then update the
   * interface magnetic fields using CT for a half time step.
   */

    integrate_emf3_corner(pG);


    for (j=js-1; j<=je+1; j++) {
      for (i=is-1; i<=ie+1; i++) {
        B1_x1Face[j][i] -= hdtodx2*(emf3[j+1][i  ] - emf3[j][i]);
        B2_x2Face[j][i] += hdtodx1*(emf3[j  ][i+1] - emf3[j][i]);
      }
      B1_x1Face[j][iu] -= hdtodx2*(emf3[j+1][iu] - emf3[j][iu]);
    }
    for (i=is-1; i<=ie+1; i++) {
      B2_x2Face[ju][i] += hdtodx1*(emf3[ju][i+1] - emf3[ju][i]);
    }


  /*--- Step 5a ------------------------------------------------------------------
   * Correct the L/R states at x1-interfaces using transverse flux-gradients in
   * the x2-direction for 0.5*dt using x2-fluxes computed in Step 2e.
   * Since the fluxes come from an x2-sweep, (x,y,z) on RHS -> (z,x,y) on LHS */


    for (j=js-1; j<=je+1; j++) {
      for (i=is-1; i<=iu; i++) {
        Ul_x1Face[j][i].d  -= hdtodx2*(x2Flux[j+1][i-1].d  - x2Flux[j][i-1].d );
        Ul_x1Face[j][i].Mx -= hdtodx2*(x2Flux[j+1][i-1].Mz - x2Flux[j][i-1].Mz);
        Ul_x1Face[j][i].My -= hdtodx2*(x2Flux[j+1][i-1].Mx - x2Flux[j][i-1].Mx);
        Ul_x1Face[j][i].Mz -= hdtodx2*(x2Flux[j+1][i-1].My - x2Flux[j][i-1].My);
        Ul_x1Face[j][i].E  -= hdtodx2*(x2Flux[j+1][i-1].E  - x2Flux[j][i-1].E );
        Ul_x1Face[j][i].Bz -= hdtodx2*(x2Flux[j+1][i-1].By - x2Flux[j][i-1].By);

        Ur_x1Face[j][i].d  -= hdtodx2*(x2Flux[j+1][i  ].d  - x2Flux[j][i  ].d );
        Ur_x1Face[j][i].Mx -= hdtodx2*(x2Flux[j+1][i  ].Mz - x2Flux[j][i  ].Mz);
        Ur_x1Face[j][i].My -= hdtodx2*(x2Flux[j+1][i  ].Mx - x2Flux[j][i  ].Mx);
        Ur_x1Face[j][i].Mz -= hdtodx2*(x2Flux[j+1][i  ].My - x2Flux[j][i  ].My);
        Ur_x1Face[j][i].E  -= hdtodx2*(x2Flux[j+1][i  ].E  - x2Flux[j][i  ].E );
        Ur_x1Face[j][i].Bz -= hdtodx2*(x2Flux[j+1][i  ].By - x2Flux[j][i  ].By);
      }
    }


  /*--- Step 5b ------------------------------------------------------------------
   * Add the "MHD source terms" to the x2 (conservative) flux gradient.
   */


    for (j=js-1; j<=je+1; j++) {
      for (i=is-1; i<=iu; i++) {
        dbx = pG->B1i[j][i] - pG->B1i[j][i-1];
        B1 = pG->U[j][i-1].B1c;
        B2 = pG->U[j][i-1].B2c;
        B3 = pG->U[j][i-1].B3c;
        V3 = pG->U[j][i-1].M3/pG->U[j][i-1].d;

        Ul_x1Face[j][i].Mx += hdtodx1*B1*dbx;
        Ul_x1Face[j][i].My += hdtodx1*B2*dbx;
        Ul_x1Face[j][i].Mz += hdtodx1*B3*dbx;
        Ul_x1Face[j][i].Bz += hdtodx1*V3*dbx;
        Ul_x1Face[j][i].E  += hdtodx1*B3*V3*dbx;

        dbx = pG->B1i[j][i+1] - pG->B1i[j][i];
        B1 = pG->U[j][i].B1c;
        B2 = pG->U[j][i].B2c;
        B3 = pG->U[j][i].B3c;
        V3 = pG->U[j][i].M3/pG->U[j][i].d;

        Ur_x1Face[j][i].Mx += hdtodx1*B1*dbx;
        Ur_x1Face[j][i].My += hdtodx1*B2*dbx;
        Ur_x1Face[j][i].Mz += hdtodx1*B3*dbx;
        Ur_x1Face[j][i].Bz += hdtodx1*V3*dbx;
        Ur_x1Face[j][i].E  += hdtodx1*B3*V3*dbx;
      }
    }


  /*--- Step 6a ------------------------------------------------------------------
   * Correct the L/R states at x2-interfaces using transverse flux-gradients in
   * the x1-direction for 0.5*dt using x1-fluxes computed in Step 1e.
   * Since the fluxes come from an x1-sweep, (x,y,z) on RHS -> (y,z,x) on LHS */


    for (j=js-1; j<=ju; j++) {
      for (i=is-1; i<=ie+1; i++) {
        Ul_x2Face[j][i].d  -= hdtodx1*(x1Flux[j-1][i+1].d  - x1Flux[j-1][i].d );
        Ul_x2Face[j][i].Mx -= hdtodx1*(x1Flux[j-1][i+1].My - x1Flux[j-1][i].My);
        Ul_x2Face[j][i].My -= hdtodx1*(x1Flux[j-1][i+1].Mz - x1Flux[j-1][i].Mz);
        Ul_x2Face[j][i].Mz -= hdtodx1*(x1Flux[j-1][i+1].Mx - x1Flux[j-1][i].Mx);
        Ul_x2Face[j][i].E  -= hdtodx1*(x1Flux[j-1][i+1].E  - x1Flux[j-1][i].E );
        Ul_x2Face[j][i].By -= hdtodx1*(x1Flux[j-1][i+1].Bz - x1Flux[j-1][i].Bz);

        Ur_x2Face[j][i].d  -= hdtodx1*(x1Flux[j  ][i+1].d  - x1Flux[j  ][i].d );
        Ur_x2Face[j][i].Mx -= hdtodx1*(x1Flux[j  ][i+1].My - x1Flux[j  ][i].My);
        Ur_x2Face[j][i].My -= hdtodx1*(x1Flux[j  ][i+1].Mz - x1Flux[j  ][i].Mz);
        Ur_x2Face[j][i].Mz -= hdtodx1*(x1Flux[j  ][i+1].Mx - x1Flux[j  ][i].Mx);
        Ur_x2Face[j][i].E  -= hdtodx1*(x1Flux[j  ][i+1].E  - x1Flux[j  ][i].E );
        Ur_x2Face[j][i].By -= hdtodx1*(x1Flux[j][i+1].Bz - x1Flux[j][i].Bz);
      }
    }


  /*--- Step 6b ------------------------------------------------------------------
   * Add the "MHD source terms" to the x1 (conservative) flux gradient.
   */


    for (j=js-1; j<=ju; j++) {
      for (i=is-1; i<=ie+1; i++) {
        dby = pG->B2i[j][i] - pG->B2i[j-1][i];
        B1 = pG->U[j-1][i].B1c;
        B2 = pG->U[j-1][i].B2c;
        B3 = pG->U[j-1][i].B3c;
        V3 = pG->U[j-1][i].M3/pG->U[j-1][i].d;

        Ul_x2Face[j][i].Mz += hdtodx2*B1*dby;
        Ul_x2Face[j][i].Mx += hdtodx2*B2*dby;
        Ul_x2Face[j][i].My += hdtodx2*B3*dby;
        Ul_x2Face[j][i].By += hdtodx2*V3*dby;
        Ul_x2Face[j][i].E  += hdtodx2*B3*V3*dby;

        dby = pG->B2i[j+1][i] - pG->B2i[j][i];
        B1 = pG->U[j][i].B1c;
        B2 = pG->U[j][i].B2c;
        B3 = pG->U[j][i].B3c;
        V3 = pG->U[j][i].M3/pG->U[j][i].d;

        Ur_x2Face[j][i].Mz += hdtodx2*B1*dby;
        Ur_x2Face[j][i].Mx += hdtodx2*B2*dby;
        Ur_x2Face[j][i].My += hdtodx2*B3*dby;
        Ur_x2Face[j][i].By += hdtodx2*V3*dby;
        Ur_x2Face[j][i].E  += hdtodx2*B3*V3*dby;
      }
    }


  /*--- Step 7 ------------------------------------------------------------------
   * Calculate the cell centered value of emf_3 at t^{n+1/2}, needed by CT
   * algorithm to integrate emf to corner in step 10
   */


    if (dhalf != NULL){
      for (j=js-1; j<=je+1; j++) {
        for (i=is-1; i<=ie+1; i++) {
          dhalf[j][i] = pG->U[j][i].d
            - hdtodx1*(x1Flux[j  ][i+1].d - x1Flux[j][i].d)
            - hdtodx2*(x2Flux[j+1][i  ].d - x2Flux[j][i].d);
        }
      }
    }

    for (j=js-1; j<=je+1; j++) {
      for (i=is-1; i<=ie+1; i++) {
        cc_pos(pG,i,j,&x1,&x2/*,&x3*/);

        d  = dhalf[j][i];

        M1 = pG->U[j][i].M1
          - hdtodx1*(x1Flux[j][i+1].Mx - x1Flux[j][i].Mx)
          - hdtodx2*(x2Flux[j+1][i].Mz - x2Flux[j][i].Mz);

        M2 = pG->U[j][i].M2
          - hdtodx1*(x1Flux[j][i+1].My - x1Flux[j][i].My)
          - hdtodx2*(x2Flux[j+1][i].Mx - x2Flux[j][i].Mx);

        B1c = 0.5*(B1_x1Face[j][i] + B1_x1Face[j][i+1]);
        B2c = 0.5*(B2_x2Face[j][i] + B2_x2Face[j+1][i]);

        emf3_cc[j][i] = (B1c*M2 - B2c*M1)/d;
      }
    }

/*--- Step 8b ------------------------------------------------------------------
 * Compute x1-fluxes from corrected L/R states.
 */


  for (j=js-1; j<=je+1; j++) {
	for (i=is; i<=ie+1; i++) {

      Cons1D_to_Prim1D(&Ul_x1Face[j][i],&Wl[j][i],&B1_x1Face[j][i]);
      Cons1D_to_Prim1D(&Ur_x1Face[j][i],&Wr[j][i],&B1_x1Face[j][i]);

      flux_roe(Ul_x1Face[j][i],Ur_x1Face[j][i],Wl[j][i],Wr[j][i],
                 B1_x1Face[j][i],&x1Flux[j][i]);
    }
  }


/*--- Step 8c ------------------------------------------------------------------
 * Compute x2-fluxes from corrected L/R states.
 */


  for (j=js; j<=je+1; j++) {
    for (i=is-1; i<=ie+1; i++) {

      Cons1D_to_Prim1D(&Ul_x2Face[j][i],&Wl[j][i],&B2_x2Face[j][i]);
      Cons1D_to_Prim1D(&Ur_x2Face[j][i],&Wr[j][i],&B2_x2Face[j][i]);

      flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[j][i],Wr[j][i],
                 B2_x2Face[j][i],&x2Flux[j][i]);
    }
  }


  /*--- Step 9 -------------------------------------------------------------------
   * Integrate emf3^{n+1/2} to the grid cell corners and then update the
   * interface magnetic fields using CT for a full time step.
   */


    integrate_emf3_corner(pG);


    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pG->B1i[j][i] -= dtodx2*(emf3[j+1][i  ] - emf3[j][i]);
        pG->B2i[j][i] += dtodx1*(emf3[j  ][i+1] - emf3[j][i]);
      }
      pG->B1i[j][ie+1] -= dtodx2*(emf3[j+1][ie+1] - emf3[j][ie+1]);
    }
    for (i=is; i<=ie; i++) {
      pG->B2i[je+1][i] += dtodx1*(emf3[je+1][i+1] - emf3[je+1][i]);
    }

  /*--- Step 11a -----------------------------------------------------------------
   * Update cell-centered variables in pG using x1-fluxes
   */

    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pG->U[j][i].d  -= dtodx1*(x1Flux[j][i+1].d  - x1Flux[j][i].d );
        pG->U[j][i].M1 -= dtodx1*(x1Flux[j][i+1].Mx - x1Flux[j][i].Mx);
        pG->U[j][i].M2 -= dtodx1*(x1Flux[j][i+1].My - x1Flux[j][i].My);
        pG->U[j][i].M3 -= dtodx1*(x1Flux[j][i+1].Mz - x1Flux[j][i].Mz);
        pG->U[j][i].E  -= dtodx1*(x1Flux[j][i+1].E  - x1Flux[j][i].E );
        pG->U[j][i].B2c -= dtodx1*(x1Flux[j][i+1].By - x1Flux[j][i].By);
        pG->U[j][i].B3c -= dtodx1*(x1Flux[j][i+1].Bz - x1Flux[j][i].Bz);
      }
    }


  /*--- Step 11b -----------------------------------------------------------------
   * Update cell-centered variables in pG using x2-fluxes
   */

    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pG->U[j][i].d  -= dtodx2*(x2Flux[j+1][i].d  - x2Flux[j][i].d );
        pG->U[j][i].M1 -= dtodx2*(x2Flux[j+1][i].Mz - x2Flux[j][i].Mz);
        pG->U[j][i].M2 -= dtodx2*(x2Flux[j+1][i].Mx - x2Flux[j][i].Mx);
        pG->U[j][i].M3 -= dtodx2*(x2Flux[j+1][i].My - x2Flux[j][i].My);
        pG->U[j][i].E  -= dtodx2*(x2Flux[j+1][i].E  - x2Flux[j][i].E );
        pG->U[j][i].B3c -= dtodx2*(x2Flux[j+1][i].By - x2Flux[j][i].By);
        pG->U[j][i].B1c -= dtodx2*(x2Flux[j+1][i].Bz - x2Flux[j][i].Bz);
      }
    }


  /*--- Step 13 ------------------------------------------------------------------
   * LAST STEP!
   * Set cell centered magnetic fields to average of updated face centered fields.
   */

    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {

        pG->U[j][i].B1c =0.5*(pG->B1i[j][i]+pG->B1i[j][i+1]);
        pG->U[j][i].B2c =0.5*(pG->B2i[j][i]+pG->B2i[j+1][i]);
  /* Set the 3-interface magnetic field equal to the cell center field. */
        pG->B3i[j][i] = pG->U[j][i].B3c;
      }
    }

  return;
}


/*----------------------------------------------------------------------------*/
/* integrate_destruct_2d:  Free temporary integration arrays */

void integrate_destruct_2d(void)
{
  if (emf3    != NULL) free_2d_array(emf3);
  if (emf3_cc != NULL) free_2d_array(emf3_cc);
  if (Bxc != NULL) free_2d_array(Bxc);
  if (Bxi != NULL) free_2d_array(Bxi);
  if (B1_x1Face != NULL) free_2d_array(B1_x1Face);
  if (B2_x2Face != NULL) free_2d_array(B2_x2Face);

  if (U1d      != NULL) free_2d_array(U1d);
  if (Ul       != NULL) free(Ul);
  if (Ur       != NULL) free(Ur);
  if (W        != NULL) free_2d_array(W);
  if (Wl       != NULL) free_2d_array(Wl);
  if (Wr       != NULL) free_2d_array(Wr);

  if (Ul_x1Face != NULL) free_2d_array(Ul_x1Face);
  if (Ur_x1Face != NULL) free_2d_array(Ur_x1Face);
  if (Ul_x2Face != NULL) free_2d_array(Ul_x2Face);
  if (Ur_x2Face != NULL) free_2d_array(Ur_x2Face);
  if (x1Flux    != NULL) free_2d_array(x1Flux);
  if (x2Flux    != NULL) free_2d_array(x2Flux);
  if (dhalf     != NULL) free_2d_array(dhalf);

  return;
}

/*----------------------------------------------------------------------------*/
/* integrate_init_2d:    Allocate temporary integration arrays */

void integrate_init_2d(int nx1, int nx2)
{
  int nmax;
  int Nx1 = nx1 + 2*nghost;
  int Nx2 = nx2 + 2*nghost;
  nmax = MAX(Nx1,Nx2);

  if ((emf3 = (Real**)calloc_2d_array(Nx2, Nx1, sizeof(Real))) == NULL)
    goto on_error;
  if ((emf3_cc = (Real**)calloc_2d_array(Nx2, Nx1, sizeof(Real))) == NULL)
    goto on_error;

  if ((Bxc = (Real**)calloc_2d_array(nmax, nmax, sizeof(Real))) == NULL) goto on_error;
  if ((Bxi = (Real**)calloc_2d_array(nmax, nmax, sizeof(Real))) == NULL) goto on_error;

  if ((B1_x1Face = (Real**)calloc_2d_array(Nx2, Nx1, sizeof(Real))) == NULL)
    goto on_error;
  if ((B2_x2Face = (Real**)calloc_2d_array(Nx2, Nx1, sizeof(Real))) == NULL)
    goto on_error;

  if ((U1d =      (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL) goto on_error;
  if ((Ul  =      (Cons1D*)malloc(nmax*sizeof(Cons1D))) == NULL) goto on_error;
  if ((Ur  =      (Cons1D*)malloc(nmax*sizeof(Cons1D))) == NULL) goto on_error;
  if ((W  =      (Prim1D**)calloc_2d_array(nmax, nmax, sizeof(Prim1D))) == NULL) goto on_error;
  if ((Wl =      (Prim1D**)calloc_2d_array(nmax, nmax, sizeof(Prim1D))) == NULL) goto on_error;
  if ((Wr =      (Prim1D**)calloc_2d_array(nmax, nmax, sizeof(Prim1D))) == NULL) goto on_error;

  if ((Ul_x1Face = (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL)
    goto on_error;
  if ((Ur_x1Face = (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL)
    goto on_error;
  if ((Ul_x2Face = (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL)
    goto on_error;
  if ((Ur_x2Face = (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL)
    goto on_error;

  if ((x1Flux    = (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL)
    goto on_error;
  if ((x2Flux    = (Cons1D**)calloc_2d_array(Nx2, Nx1, sizeof(Cons1D))) == NULL)
    goto on_error;

  if ((dhalf = (Real**)calloc_2d_array(Nx2, Nx1, sizeof(Real))) == NULL)
    goto on_error;

  return;

  on_error:
  //integrate_destruct();
  integrate_destruct_2d();
  ath_error("[integrate_init]: malloc returned a NULL pointer\n");
}



/*=========================== PRIVATE FUNCTIONS ==============================*/

/*----------------------------------------------------------------------------*/
/* integrate_emf3_corner:  */

static void integrate_emf3_corner(Grid *pG)
{
  int i,is,ie,j,js,je;
  Real emf_l1, emf_r1, emf_l2, emf_r2;

  is = pG->is;   ie = pG->ie;
  js = pG->js;   je = pG->je;

/* NOTE: The x1-Flux of B2 is -E3.  The x2-Flux of B1 is +E3. */
  for (j=js-1; j<=je+2; j++) {
    for (i=is-1; i<=ie+2; i++) {
      if (x1Flux[j-1][i].d > 0.0) {
	emf_l2 = -x1Flux[j-1][i].By
	  + (x2Flux[j][i-1].Bz - emf3_cc[j-1][i-1]);
      }
      else if (x1Flux[j-1][i].d < 0.0) {
	emf_l2 = -x1Flux[j-1][i].By
	  + (x2Flux[j][i].Bz - emf3_cc[j-1][i]);

      } else {
	emf_l2 = -x1Flux[j-1][i].By
	  + 0.5*(x2Flux[j][i-1].Bz - emf3_cc[j-1][i-1] + 
		 x2Flux[j][i  ].Bz - emf3_cc[j-1][i  ] );
      }

      if (x1Flux[j][i].d > 0.0) {
	emf_r2 = -x1Flux[j][i].By 
	  + (x2Flux[j][i-1].Bz - emf3_cc[j][i-1]);
      }
      else if (x1Flux[j][i].d < 0.0) {
	emf_r2 = -x1Flux[j][i].By 
	  + (x2Flux[j][i].Bz - emf3_cc[j][i]);

      } else {
	emf_r2 = -x1Flux[j][i].By 
	  + 0.5*(x2Flux[j][i-1].Bz - emf3_cc[j][i-1] + 
		 x2Flux[j][i  ].Bz - emf3_cc[j][i  ] );
      }

      if (x2Flux[j][i-1].d > 0.0) {
	emf_l1 = x2Flux[j][i-1].Bz
	  + (-x1Flux[j-1][i].By - emf3_cc[j-1][i-1]);
      }
      else if (x2Flux[j][i-1].d < 0.0) {
	emf_l1 = x2Flux[j][i-1].Bz 
          + (-x1Flux[j][i].By - emf3_cc[j][i-1]);
      } else {
	emf_l1 = x2Flux[j][i-1].Bz
	  + 0.5*(-x1Flux[j-1][i].By - emf3_cc[j-1][i-1]
		 -x1Flux[j  ][i].By - emf3_cc[j  ][i-1] );
      }

      if (x2Flux[j][i].d > 0.0) {
	emf_r1 = x2Flux[j][i].Bz
	  + (-x1Flux[j-1][i].By - emf3_cc[j-1][i]);
      }
      else if (x2Flux[j][i].d < 0.0) {
	emf_r1 = x2Flux[j][i].Bz
	  + (-x1Flux[j][i].By - emf3_cc[j][i]);
      } else {
	emf_r1 = x2Flux[j][i].Bz
	  + 0.5*(-x1Flux[j-1][i].By - emf3_cc[j-1][i]
		 -x1Flux[j  ][i].By - emf3_cc[j  ][i] );
      }

      emf3[j][i] = 0.25*(emf_l1 + emf_r1 + emf_l2 + emf_r2);
    }
  }

  return;
}
