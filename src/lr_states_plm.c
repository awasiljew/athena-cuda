/* DONE - OK */

/**
 * Modified by awasiljew -> awasiljew@gmail.com 
 * for simplicity to CUDA assume that:
 *  MHD, adiabatic equation of state, no H_CORRECTION, NSCALARS =0, NO_SELF_GRAVITY, SELF_GRAVITY_NONE,
 *  GET_FLUXES=flux_roe - Roe Flux, StaticGravPot = NULL, SECOND ORDER.
 * NSCALARS=0,
 * NWAVE=7, NVAR=8
 */


//#include "copyright.h"
/*==============================================================================
 * FILE: lr_states_plm.c
 *
 * PURPOSE: Second order (piecewise linear) spatial reconstruction using
 *   characteristic interpolation in the primitive variables.  A time-evolution
 *   (characteristic tracing) step is used to interpolate interface values to
 *   the half time level {n+1/2}, unless the unsplit integrator in 3D is VL. 
 *
 * NOTATION: 
 *   W_{L,i-1/2} is reconstructed value on the left-side of interface at i-1/2
 *   W_{R,i-1/2} is reconstructed value on the right-side of interface at i-1/2
 *
 *   The L- and R-states at the left-interface in each cell are indexed i.
 *   W_{L,i-1/2} is denoted by Wl[i  ];   W_{R,i-1/2} is denoted by Wr[i  ]
 *   W_{L,i+1/2} is denoted by Wl[i+1];   W_{R,i+1/2} is denoted by Wr[i+1]
 *
 *   Internally, in this routine, Wlv and Wrv are the reconstructed values on
 *   the left-and right-side of cell center.  Thus (see Step 8),
 *     W_{L,i-1/2} = Wrv(i-1);  W_{R,i-1/2} = Wlv(i)
 *
 * CONTAINS PUBLIC FUNCTIONS:
 *   lr_states()          - computes L/R states
 *   lr_states_init()     - initializes memory for static global arrays
 *   lr_states_destruct() - frees memory for static global arrays
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

static Real **pW=NULL;


/*----------------------------------------------------------------------------*/
/* lr_states:
 * Input Arguments:
 *   W = PRIMITIVE variables at cell centers along 1-D slice
 *   Bxc = B in direction of slice at cell center
 *   dt = timestep;   dtodx = dt/dx
 *   il,iu = lower and upper indices of zone centers in slice
 * W and Bxc must be initialized over [il-2:iu+2]
 *
 * Output Arguments:
 *   Wl,Wr = L/R-states of PRIMITIVE variables at interfaces over [il:iu+1]
 */

void lr_states(const Prim1D W[], const Real Bxc[],
               const Real dt, const Real dtodx, const int il, const int iu,
               Prim1D Wl[], Prim1D Wr[])
{
  int i,n,m;
  Real pb,lim_slope1,lim_slope2,qa,qb,qc,qx;
  Real ev[NWAVE],rem[NWAVE][NWAVE],lem[NWAVE][NWAVE];
  Real dWc[NWAVE],dWl[NWAVE];
  Real dWr[NWAVE],dWg[NWAVE];
  Real dac[NWAVE],dal[NWAVE];
  Real dar[NWAVE],dag[NWAVE],da[NWAVE];
  Real Wlv[NWAVE],Wrv[NWAVE];
  Real dW[NWAVE],dWm[NWAVE];
  Real *pWl, *pWr;

  // il = is-1
  // iu = ie+1

/* Zero eigenmatrices, set pointer to primitive variables */
  for (n=0; n<NWAVE; n++) {
    for (m=0; m<NWAVE; m++) {
      rem[n][m] = 0.0;
      lem[n][m] = 0.0;
    }
  }
  /**
   * W vector is a vector of structure Prim1D - primary variables
   */
  for (i=il-2; i<=iu+2; i++) pW[i] = (Real*)&(W[i]);

/*========================== START BIG LOOP OVER i =======================*/
  for (i=il-1; i<=iu+1; i++) {

	  // Loop over i == is-2 .. ie+2

/*--- Step 1. ------------------------------------------------------------------
 * Compute eigensystem in primitive variables.  */

    esys_prim_adb_mhd(W[i].d,W[i].Vx,W[i].P,Bxc[i],W[i].By,W[i].Bz,ev,rem,lem);

/*--- Step 2. ------------------------------------------------------------------
 * Compute centered, L/R, and van Leer differences of primitive variables
 * Note we access contiguous array elements by indexing pointers for speed */

    for (n=0; n<NWAVE; n++) {
      dWc[n] = pW[i+1][n] - pW[i-1][n];
      dWl[n] = pW[i][n]   - pW[i-1][n];
      dWr[n] = pW[i+1][n] - pW[i][n];
      if (dWl[n]*dWr[n] > 0.0) {
        dWg[n] = 2.0*dWl[n]*dWr[n]/(dWl[n]+dWr[n]);
      } else {
        dWg[n] = 0.0;
      }
    }

/*--- Step 3. ------------------------------------------------------------------
 * Project differences in primitive variables along characteristics */

    for (n=0; n<NWAVE; n++) {
      dac[n] = lem[n][0]*dWc[0];
      dal[n] = lem[n][0]*dWl[0];
      dar[n] = lem[n][0]*dWr[0];
      dag[n] = lem[n][0]*dWg[0];
      for (m=1; m<NWAVE; m++) {
	dac[n] += lem[n][m]*dWc[m];
	dal[n] += lem[n][m]*dWl[m];
	dar[n] += lem[n][m]*dWr[m];
	dag[n] += lem[n][m]*dWg[m];
      }
    }

/*--- Step 4. ------------------------------------------------------------------
 * Apply monotonicity constraints to characteristic projections */

    for (n=0; n<NWAVE; n++) {
      da[n] = 0.0;
      if (dal[n]*dar[n] > 0.0) {
        lim_slope1 = MIN(    fabs(dal[n]),fabs(dar[n]));
        lim_slope2 = MIN(0.5*fabs(dac[n]),fabs(dag[n]));
        da[n] = SIGN(dac[n])*MIN(2.0*lim_slope1,lim_slope2);
      }
    }

/*--- Step 5. ------------------------------------------------------------------
 * Project monotonic slopes in characteristic back to primitive variables  */

    for (n=0; n<NWAVE; n++) {
      dWm[n] = da[0]*rem[n][0];
      for (m=1; m<NWAVE; m++) {
        dWm[n] += da[m]*rem[n][m];
      }
    }

/*--- Step 7. ------------------------------------------------------------------
 * Compute L/R values, ensure they lie between neighboring cell-centered vals */

    for (n=0; n<NWAVE; n++) {
      Wlv[n] = pW[i][n] - 0.5*dWm[n];
      Wrv[n] = pW[i][n] + 0.5*dWm[n];
    }

    for (n=0; n<NWAVE; n++) {
      Wlv[n] = MAX(MIN(pW[i][n],pW[i-1][n]),Wlv[n]);
      Wlv[n] = MIN(MAX(pW[i][n],pW[i-1][n]),Wlv[n]);
      Wrv[n] = MAX(MIN(pW[i][n],pW[i+1][n]),Wrv[n]);
      Wrv[n] = MIN(MAX(pW[i][n],pW[i+1][n]),Wrv[n]);
    }

    for (n=0; n<NWAVE; n++) {
      dW[n] = Wrv[n] - Wlv[n];
    }

/*--- Step 8. ------------------------------------------------------------------
 * Integrate linear interpolation function over domain of dependence defined by
 * max(min) eigenvalue
 */

    pWl = (Real *) &(Wl[i+1]);
    pWr = (Real *) &(Wr[i]);

    qx = 0.5*MAX(ev[NWAVE-1],0.0)*dtodx;
    for (n=0; n<NWAVE; n++) {
      pWl[n] = Wrv[n] - qx*dW[n];
    }

    qx = -0.5*MIN(ev[0],0.0)*dtodx;
    for (n=0; n<NWAVE; n++) {
      pWr[n] = Wlv[n] + qx*dW[n];
    }


/*--- Step 9. ------------------------------------------------------------------
 * Then subtract amount of each wave n that does not reach the interface
 * during timestep (CW eqn 3.5ff).  For HLL fluxes, must subtract waves that
 * move in both directions.
 */

    for (n=0; n<NWAVE; n++) {
      if (ev[n] > 0.) {
	qa  = 0.0;
	for (m=0; m<NWAVE; m++) {
	  qa += lem[n][m]*0.5*dtodx*(ev[NWAVE-1]-ev[n])*dW[m];
	}
	for (m=0; m<NWAVE; m++) pWl[m] += qa*rem[m][n];
      }
    }

    for (n=0; n<NWAVE; n++) {
      if (ev[n] < 0.) {
        qa = 0.0;
        for (m=0; m<NWAVE; m++) {
          qa += lem[n][m]*0.5*dtodx*(ev[0]-ev[n])*dW[m];
        }
        for (m=0; m<NWAVE; m++) pWr[m] += qa*rem[m][n];
	qa  = 0.0;
	for (m=0; m<NWAVE; m++) {
	  qa += lem[n][m]*0.5*dtodx*(ev[n]-ev[NWAVE-1])*dW[m];
	}
	for (m=0; m<NWAVE; m++) pWl[m] -= qa*rem[m][n];
      }
    }

/* Wave subtraction for advected quantities */
    for (n=NWAVE; n<NWAVE; n++) {
      if (W[i].Vx > 0.) {
        pWl[n] += 0.5*dtodx*(ev[NWAVE-1]-W[i].Vx)*dW[n];
      } else if (W[i].Vx < 0.) {
        pWr[n] += 0.5*dtodx*(ev[0]-W[i].Vx)*dW[n];
      }
    }

//#ifdef __DEVICE_EMULATION__

  /* Check */
  //if(fabs(Wl[i+1].By) < 1.0e-22) {
//	  printf("HOST BY LESS THAN TINY NUMBER!!! %d %e\n", i+1, Wl[i+1].By);
  //}

//#endif


  } /*===================== END BIG LOOP OVER i ===========================*/

  return;
}

/*----------------------------------------------------------------------------*/
/* lr_states_init:  Allocate enough memory for work arrays */

void lr_states_init(int nx1, int nx2/*, int nx3*/)
{
  int i, nmax;
  nmax =  nx1 > nx2  ? nx1 : nx2;
  nmax += 2*nghost;

  if ((pW = (Real**)malloc(nmax*sizeof(Real*))) == NULL) goto on_error;

  return;
  on_error:
    lr_states_destruct();
    ath_error("[lr_states_init]: malloc returned a NULL pointer\n");
}

/*----------------------------------------------------------------------------*/
/* lr_states_destruct:  Free memory used by work arrays */

void lr_states_destruct(void)
{
  if (pW != NULL) free(pW);
  return;
}
