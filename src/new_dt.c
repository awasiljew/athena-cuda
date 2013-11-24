/* DONE - OK */

// #include "copyright.h"
/*==============================================================================
 * FILE: new_dt.c
 *
 * PURPOSE: Computes timestep using CFL condition on cell-centered velocities
 *   and sound speed, and Alfven speed from face-centered B.
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   new_dt()  - computes dt
 *   sync_dt() - synchronizes dt across all MPI patches
 *============================================================================*/

#include <stdio.h>
#include <math.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

/*----------------------------------------------------------------------------*/
/* new_dt:  */

void new_dt(Grid *pGrid)
{
  //printf("Start new DT\n");
  int i,j; //,k;
  Real di,v1,v2,v3,qsq,p,asq,cf1sq,cf2sq,cf3sq,max_dti=0.0;
//#ifdef MHD
  Real b1,b2,b3,bsq,tsum,tdif;
//#endif /* MHD */

//  for (k=pGrid->ks; k<=pGrid->ke; k++) {
  for (j=pGrid->js; j<=pGrid->je; j++) {
    for (i=pGrid->is; i<=pGrid->ie; i++) {
      di = 1.0/(pGrid->U[j][i].d);
      v1 = pGrid->U[j][i].M1*di;
      v2 = pGrid->U[j][i].M2*di;
      v3 = pGrid->U[j][i].M3*di;
      qsq = v1*v1 + v2*v2 + v3*v3;

// #ifdef MHD

/* Use maximum of face-centered fields (always larger than cell-centered B) */
      b1 = pGrid->U[j][i].B1c 
        + fabs((double)(pGrid->B1i[j][i] - pGrid->U[j][i].B1c));
      b2 = pGrid->U[j][i].B2c 
        + fabs((double)(pGrid->B2i[j][i] - pGrid->U[j][i].B2c));
      b3 = pGrid->U[j][i].B3c 
        + fabs((double)(pGrid->B3i[j][i] - pGrid->U[j][i].B3c));
      bsq = b1*b1 + b2*b2 + b3*b3;
/* compute sound speed squared */
      p = MAX(Gamma_1*(pGrid->U[j][i].E - 0.5*pGrid->U[j][i].d*qsq
              - 0.5*bsq), TINY_NUMBER);
      asq = Gamma*p*di;

/* compute fast magnetosonic speed squared in each direction */
      tsum = bsq*di + asq;
      tdif = bsq*di - asq;
      cf1sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b2*b2+b3*b3)*di));
      cf2sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b1*b1+b3*b3)*di));
      cf3sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b1*b1+b2*b2)*di));


/* compute sound speed squared */
// #ifdef ADIABATIC
      p = MAX(Gamma_1*(pGrid->U[j][i].E - 0.5*pGrid->U[j][i].d*qsq),
              TINY_NUMBER);
      asq = Gamma*p*di;
// #else
//      asq = Iso_csound2;
// #endif /* ADIABATIC */
/* compute fast magnetosonic speed squared in each direction */
      cf1sq = asq;
      cf2sq = asq;
      cf3sq = asq;

// #endif /* MHD */

/* compute maximum inverse of dt (corresponding to minimum dt) */
      if (pGrid->Nx1 > 1)
        max_dti = MAX(max_dti,(fabs(v1)+sqrt((double)cf1sq))/pGrid->dx1);
      if (pGrid->Nx2 > 1)
        max_dti = MAX(max_dti,(fabs(v2)+sqrt((double)cf2sq))/pGrid->dx2);
   //   if (pGrid->Nx3 > 1)
   //     max_dti = MAX(max_dti,(fabs(v3)+sqrt((double)cf3sq))/pGrid->dx3);
//
  //  }
  }}

/* new timestep.  Limit increase to 2x old value */
  if (pGrid->nstep == 0) {
    pGrid->dt = CourNo/max_dti;
  } else {
    pGrid->dt = MIN(2.0*pGrid->dt, CourNo/max_dti);
  }

  //printf("New DT calculated %e\n", pGrid->dt);

  return;
}

