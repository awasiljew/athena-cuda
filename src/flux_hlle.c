/* DONE - OK  */

/**
 * Modified by awasiljew -> awasiljew@gmail.com
 * Check that the density and pressure in the intermediate states are positive -> if not from reo seolver we
 * invoke below function.
 * * for simplicity to CUDA assume that:
 *  MHD, adiabatic equation of state, no H_CORRECTION, NSCALARS =0, NO_SELF_GRAVITY, SELF_GRAVITY_NONE,
 *  GET_FLUXES=flux_roe - Roe Flux, StaticGravPot = NULL, SECOND ORDER.
 * NSCALARS=0,
 * NWAVE=7, NVAR=8
 */


//#include "copyright.h"
/*==============================================================================
 * FILE: flux_hlle.c
 *
 * PURPOSE: Computes 1D fluxes using the Harten-Lax-van Leer (HLLE) Riemann
 *   solver.  This flux is very diffusive, especially for contacts, and so it
 *   is not recommended for use in applications.  However, as shown by Einfeldt
 *   et al.(1991), it is positively conservative, so it is a useful option when
 *   other approximate solvers fail and/or when extra dissipation is needed.
 *
 * REFERENCES:
 *   E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics",
 *   2nd ed., Springer-Verlag, Berlin, (1999) chpt. 10.
 *
 *   Einfeldt et al., "On Godunov-type methods near low densities",
 *   JCP, 92, 273 (1991)
 *
 *   A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and
 *   Godunov-type schemes for hyperbolic conservation laws",
 *   SIAM Review 25, 35-61 (1983).
 *
 *   B. Einfeldt, "On Godunov-type methods for gas dynamics",
 *   SIAM J. Numerical Analysis 25, 294-318 (1988).
 *
 * CONTAINS PUBLIC FUNCTIONS:
 *   flux_hlle()
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

/*----------------------------------------------------------------------------*/
/* flux_hlle:
 *   Input Arguments:
 *     Bxi = B in direction of 1D slice at cell interface
 *     Ul,Ur = L/R-states of CONSERVED variables at cell interface
 *   Output Arguments:
 *     pFlux = pointer to fluxes of CONSERVED variables at cell interface
 */

void flux_hlle(const Cons1D Ul, const Cons1D Ur,
               const Prim1D Wl, const Prim1D Wr, const Real Bxi, Cons1D *pFlux)
{
  Real sqrtdl,sqrtdr,isdlpdr,droe,v1roe,v2roe,v3roe,pbl=0.0,pbr=0.0;
  Real asq,vaxsq=0.0,qsq,cfsq,cfl,cfr,bp,bm,ct2=0.0,tmp;
  Real hroe;
  Real b2roe,b3roe,x,y;
  Real ev[NWAVE],al,ar;
  Real *pFl, *pFr, *pF;
/*  Prim1D Wl, Wr; */
  Cons1D Fl,Fr;
  int n;


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

  sqrtdl = sqrt((double)Wl.d);
  sqrtdr = sqrt((double)Wr.d);
  isdlpdr = 1.0/(sqrtdl + sqrtdr);

  droe  = sqrtdl*sqrtdr;
  v1roe = (sqrtdl*Wl.Vx + sqrtdr*Wr.Vx)*isdlpdr;
  v2roe = (sqrtdl*Wl.Vy + sqrtdr*Wr.Vy)*isdlpdr;
  v3roe = (sqrtdl*Wl.Vz + sqrtdr*Wr.Vz)*isdlpdr;

/* The Roe average of the magnetic field is defined differently.  */

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
 * Compute eigenvalues using Roe-averaged values, needed in step 4.
 */

 esys_roe_adb_mhd(droe,v1roe,v2roe,v3roe,hroe,Bxi,b2roe,b3roe,x,y,ev,NULL,NULL);

/*--- Step 4. ------------------------------------------------------------------
 * Compute the max and min wave speeds
 */

  asq = Gamma*Wl.P/Wl.d;
  vaxsq = Bxi*Bxi/Wl.d;
  ct2 = (Ul.By*Ul.By + Ul.Bz*Ul.Bz)/Wl.d;
  qsq = vaxsq + ct2 + asq;
  tmp = vaxsq + ct2 - asq;
  cfsq = 0.5*(qsq + sqrt((double)(tmp*tmp + 4.0*asq*ct2)));
  cfl = sqrt((double)cfsq);

  asq = Gamma*Wr.P/Wr.d; 
  vaxsq = Bxi*Bxi/Wr.d;
  ct2 = (Ur.By*Ur.By + Ur.Bz*Ur.Bz)/Wr.d;
  qsq = vaxsq + ct2 + asq;
  tmp = vaxsq + ct2 - asq;
  cfsq = 0.5*(qsq + sqrt((double)(tmp*tmp + 4.0*asq*ct2)));
  cfr = sqrt((double)cfsq);

/* take max/min of Roe eigenvalues and L/R state wave speeds */
  ar = MAX(ev[NWAVE-1],(Wr.Vx + cfr));
  al = MIN(ev[0]      ,(Wl.Vx - cfl));

  bp = MAX(ar, 0.0);
  bm = MIN(al, 0.0);

/*--- Step 5. ------------------------------------------------------------------
 * Compute L/R fluxes along the lines bm/bp: F_{L}-S_{L}U_{L}; F_{R}-S_{R}U_{R}
 */

  Fl.d  = Ul.Mx - bm*Ul.d;
  Fr.d  = Ur.Mx - bp*Ur.d;

  Fl.Mx = Ul.Mx*(Wl.Vx - bm);
  Fr.Mx = Ur.Mx*(Wr.Vx - bp);

  Fl.My = Ul.My*(Wl.Vx - bm);
  Fr.My = Ur.My*(Wr.Vx - bp);

  Fl.Mz = Ul.Mz*(Wl.Vx - bm);
  Fr.Mz = Ur.Mz*(Wr.Vx - bp);

  Fl.Mx += Wl.P;
  Fr.Mx += Wr.P;

  Fl.E  = Ul.E*(Wl.Vx - bm) + Wl.P*Wl.Vx;
  Fr.E  = Ur.E*(Wr.Vx - bp) + Wr.P*Wr.Vx;

  Fl.Mx -= 0.5*(Bxi*Bxi - SQR(Wl.By) - SQR(Wl.Bz));
  Fr.Mx -= 0.5*(Bxi*Bxi - SQR(Wr.By) - SQR(Wr.Bz));

  Fl.My -= Bxi*Wl.By;
  Fr.My -= Bxi*Wr.By;
    
  Fl.Mz -= Bxi*Wl.Bz;
  Fr.Mz -= Bxi*Wr.Bz;

  Fl.E += (pbl*Wl.Vx - Bxi*(Bxi*Wl.Vx + Wl.By*Wl.Vy + Wl.Bz*Wl.Vz));
  Fr.E += (pbr*Wr.Vx - Bxi*(Bxi*Wr.Vx + Wr.By*Wr.Vy + Wr.Bz*Wr.Vz));

  Fl.By = Wl.By*(Wl.Vx - bm) - Bxi*Wl.Vy;
  Fr.By = Wr.By*(Wr.Vx - bp) - Bxi*Wr.Vy;

  Fl.Bz = Wl.Bz*(Wl.Vx - bm) - Bxi*Wl.Vz;
  Fr.Bz = Wr.Bz*(Wr.Vx - bp) - Bxi*Wr.Vz;

  
/*--- Step 6. ------------------------------------------------------------------
 * Compute the HLLE flux at interface.
 */

  pFl = (Real *)&(Fl);
  pFr = (Real *)&(Fr);
  pF  = (Real *)pFlux;
  tmp = 0.5*(bp + bm)/(bp - bm);
  for (n=0; n<(NWAVE+NSCALARS); n++){
    pF[n] = 0.5*(pFl[n] + pFr[n]) + (pFl[n] - pFr[n])*tmp;
  }

  return;
}
