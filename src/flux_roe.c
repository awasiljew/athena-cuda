/* DONE - OK */

/**
 * Modified by awasiljew -> awasiljew@gmail.com
 * for simplicity to CUDA assume that:
 *  MHD, adiabatic equation of state, no H_CORRECTION, NSCALARS =0, NO_SELF_GRAVITY, SELF_GRAVITY_NONE,
 *  GET_FLUXES=flux_roe - Roe Flux, StaticGravPot = NULL, SECOND ORDER.
 * NSCALARS=0,
 * NWAVE=7, NVAR=8
 *
 */ 


//#include "copyright.h"
/*==============================================================================
 * FILE: flux_roe.c
 *
 * PURPOSE: Computes 1D fluxes using Roe's linearization.  When Roe's method
 * fails because of negative density or pressure in the intermediate states,
 * the fluxes are computed with the HLLE solver instead.
 *
 * REFERENCES:
 *   P. Roe, "Approximate Riemann solvers, parameter vectors, and difference
 *   schemes", JCP, 43, 357 (1981).
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   flux_roe()
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include "debug_tools_cuda.h"

/* maximum wavespeed used by H-correction, value passed from integrator */
Real etah=0.0;

/* Test the intermediate states in the approximate Riemann solution. */
#define TEST_INTERMEDIATE_STATES

/*----------------------------------------------------------------------------*/
/* flux_roe
 *   Input Arguments:
 *     Bxi = B in direction of slice at cell interface
 *     Ul,Ur = L/R-states of CONSERVED variables at cell interface
 *   Output Arguments:
 *     pFlux = pointer to fluxes of CONSERVED variables at cell interface
 */

void flux_roe(const Cons1D Ul, const Cons1D Ur,
              const Prim1D Wl, const Prim1D Wr, const Real Bxi, Cons1D *pFlux)
{
  Real sqrtdl,sqrtdr,isdlpdr,droe,v1roe,v2roe,v3roe,pbl=0.0,pbr=0.0;
  Real hroe;
  Real b2roe,b3roe,x,y;
  Real coeff[NWAVE];
  Real ev[NWAVE],rem[NWAVE][NWAVE],lem[NWAVE][NWAVE];
  Real dU[NWAVE],a[NWAVE];
#ifdef TEST_INTERMEDIATE_STATES
  Real u_inter[NWAVE],p_inter=0.0;
#endif /* TEST_INTERMEDIATE_STATES */
/*  Prim1D Wl, Wr; */
  Real *pUl, *pUr, *pFl, *pFr, *pF;
  Cons1D Fl,Fr;
  int n,m,hlle_flag;

  for (n=0; n<NWAVE; n++) {
    for (m=0; m<NWAVE; m++) {
      rem[n][m] = 0.0;
      lem[n][m] = 0.0;
    }
  }

/*--- Step 1. ------------------------------------------------------------------
 * Convert left- and right- states in conserved to primitive variables.
 */

/*
  pbl = Cons1D_to_Prim1D(&Ul,&Wl,&Bxi);
  pbr = Cons1D_to_Prim1D(&Ur,&Wr,&Bxi);
*/

/*--- Step 2. ------------------------------------------------------------------
 * Compute Roe-averaged data from left- and right-states
 */

  sqrtdl = sqrt((Real)Wl.d);
  sqrtdr = sqrt((Real)Wr.d);
  isdlpdr = 1.0/(sqrtdl + sqrtdr);

  droe  = sqrtdl*sqrtdr;
  v1roe = (sqrtdl*Wl.Vx + sqrtdr*Wr.Vx)*isdlpdr;
  v2roe = (sqrtdl*Wl.Vy + sqrtdr*Wr.Vy)*isdlpdr;
  v3roe = (sqrtdl*Wl.Vz + sqrtdr*Wr.Vz)*isdlpdr;

/* The Roe average of the magnetic field is defined differently  */

  b2roe = (sqrtdr*Wl.By + sqrtdl*Wr.By)*isdlpdr;
  b3roe = (sqrtdr*Wl.Bz + sqrtdl*Wr.Bz)*isdlpdr;
  x = 0.5*(SQR(Wl.By - Wr.By) + SQR(Wl.Bz - Wr.Bz))/(SQR(sqrtdl + sqrtdr));
  y = 0.5*(Wl.d + Wr.d)/droe;
  pbl = 0.5*(SQR(Bxi) + SQR(Wl.By) + SQR(Wl.Bz));
  pbr = 0.5*(SQR(Bxi) + SQR(Wr.By) + SQR(Wr.Bz));

/*
 * Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
 * rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
 */

  hroe  = ((Ul.E + Wl.P + pbl)/sqrtdl + (Ur.E + Wr.P + pbr)/sqrtdr)*isdlpdr;

/*--- Step 3. ------------------------------------------------------------------
 * Compute eigenvalues and eigenmatrices using Roe-averaged values
 */

//#ifndef ESYS_ROE_CUDA

//  if(!ESYS_ROE_CUDA_FLAG) {
	  esys_roe_adb_mhd(droe,v1roe,v2roe,v3roe,hroe,Bxi,b2roe,b3roe,x,y,ev,rem,lem);
//  } else {
//#else

//	  esys_roe_adb_mhd_cuda(droe,v1roe,v2roe,v3roe,hroe,Bxi,b2roe,b3roe,x,y,ev,rem,lem, Gamma_1, Gamma_2);
 // }

//#endif

/*--- Step 4. ------------------------------------------------------------------
 * Compute L/R fluxes 
 */

  Fl.d  = Ul.Mx;
  Fr.d  = Ur.Mx;

  Fl.Mx = Ul.Mx*Wl.Vx;
  Fr.Mx = Ur.Mx*Wr.Vx;

  Fl.My = Ul.Mx*Wl.Vy;
  Fr.My = Ur.Mx*Wr.Vy;

  Fl.Mz = Ul.Mx*Wl.Vz;
  Fr.Mz = Ur.Mx*Wr.Vz;

  Fl.Mx += Wl.P;
  Fr.Mx += Wr.P;

  Fl.E  = (Ul.E + Wl.P)*Wl.Vx;
  Fr.E  = (Ur.E + Wr.P)*Wr.Vx;

  Fl.Mx -= 0.5*(Bxi*Bxi - SQR(Wl.By) - SQR(Wl.Bz));
  Fr.Mx -= 0.5*(Bxi*Bxi - SQR(Wr.By) - SQR(Wr.Bz));

  Fl.My -= Bxi*Wl.By;
  Fr.My -= Bxi*Wr.By;
    
  Fl.Mz -= Bxi*Wl.Bz;
  Fr.Mz -= Bxi*Wr.Bz;

  Fl.E += (pbl*Wl.Vx - Bxi*(Bxi*Wl.Vx + Wl.By*Wl.Vy + Wl.Bz*Wl.Vz));
  Fr.E += (pbr*Wr.Vx - Bxi*(Bxi*Wr.Vx + Wr.By*Wr.Vy + Wr.Bz*Wr.Vz));

  Fl.By = Wl.By*Wl.Vx - Bxi*Wl.Vy;
  Fr.By = Wr.By*Wr.Vx - Bxi*Wr.Vy;


  //printf("%.15e %.15e %.15e %.15e\n", Wl.Bz, Wl.Vx, Bxi, Wl.Vz);
  Fl.Bz = Wl.Bz*Wl.Vx - Bxi*Wl.Vz;
  Fr.Bz = Wr.Bz*Wr.Vx - Bxi*Wr.Vz;


/*--- Step 5. ------------------------------------------------------------------
 * Return upwind flux if flow is supersonic
 */

  //printf("ev[0] %e\n", ev[0]);

  if(ev[0] >= 0.0){
    *pFlux = Fl;
    //printf("Left %e %e %e %e %e %e %e \n", Fl.d, Fl.Mx, Fl.My, Fl.Mz, Fl.By, Fl.Bz, Fl.E);
    return;
  }

  if(ev[NWAVE-1] <= 0.0){
    *pFlux = Fr;
    //printf("Right %e %e %e %e %e %e %e \n", Fr.d, Fr.Mx, Fr.My, Fr.Mz, Fr.By, Fr.Bz, Fr.E);
    return;
  }

/*--- Step 6. ------------------------------------------------------------------
 * Compute projection of dU onto L eigenvectors ("vector A")
 */

  pUr = (Real *)&(Ur);
  pUl = (Real *)&(Ul);

  for (n=0; n<NWAVE; n++) dU[n] = pUr[n] - pUl[n];
  for (n=0; n<NWAVE; n++) {
    a[n] = 0.0;
    for (m=0; m<NWAVE; m++) a[n] += lem[n][m]*dU[m];
  }

/*--- Step 7. ------------------------------------------------------------------
 * Check that the density and pressure in the intermediate states are positive.
 * If not, set hlle_flag=1 if d_inter<0; hlle_flag=2 if p_inter<0, get HLLE
 * fluxes, and return
 */

  hlle_flag = 0;
#ifdef TEST_INTERMEDIATE_STATES

  for (n=0; n<NWAVE; n++) u_inter[n] = pUl[n];
  for (n=0; n<NWAVE-1; n++) {
    for (m=0; m<NWAVE; m++) u_inter[m] += a[n]*rem[m][n];
    if(ev[n+1] > ev[n]) {
      if (u_inter[0] <= 0.0) {
	hlle_flag=1;
	break;
      }
      p_inter = u_inter[4] - 0.5*
	(SQR(u_inter[1])+SQR(u_inter[2])+SQR(u_inter[3]))/u_inter[0];
      p_inter -= 0.5*(SQR(u_inter[NWAVE-2])+SQR(u_inter[NWAVE-1])+SQR(Bxi));
      if (p_inter < 0.0) {
	hlle_flag=2;
	break;
      }
    }
  }

  if (hlle_flag != 0) {
	//printf("HLLE\n");
    flux_hlle(Ul,Ur,Wl,Wr,Bxi,pFlux);
    return;
  }

#endif /* TEST_INTERMEDIATE_STATES */

/*--- Step 8. ------------------------------------------------------------------
 * Compute Roe flux */

  pFl = (Real *)&(Fl);
  pFr = (Real *)&(Fr);
  pF  = (Real *)(pFlux); 

  for (m=0; m<NWAVE; m++) {
    coeff[m] = 0.5*MAX(fabs(ev[m]),etah)*a[m];
  }
  for (n=0; n<NWAVE; n++) {
    pF[n] = 0.5*(pFl[n] + pFr[n]);
    for (m=0; m<NWAVE; m++) {
      pF[n] -= coeff[m]*rem[n][m];
    }
  }

  return;
}
