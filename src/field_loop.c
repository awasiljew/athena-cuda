/* DONE - 2D field loop problem */

#include "copyright.h"
/*==============================================================================
 * FILE: field_loop.c
 *
 * PURPOSE: Problem generator for advection of a field loop test.  Can only
 *   be run in 2D or 3D.  Input parameters are:
 *      problem/rad   = radius of field loop
 *      problem/amp   = amplitude of vector potential (and therefore B)
 *      problem/vflow = flow velocity
 *   The flow is automatically set to run along the diagonal. 
 *   Various test cases are possible:
 *     (iprob=1): field loop in x1-x2 plane (cylinder in 3D)
 *     (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
 *     (iprob=3): field loop in x3-x1 plane (cylinder in 3D) 
 *     (iprob=4): rotated cylindrical field loop in 3D.
 *     (iprob=5): spherical field loop in rotated plane
 *   A sphere of passive scalar can be added to test advection of scalars.
 *
 * REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD
 *   via constrined transport", JCP, 205, 509 (2005)
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include "debug_tools_cuda.h"

/*----------------------------------------------------------------------------*/
/* problem:   */

#ifdef FIELD_LOOP

void problem(Grid *pGrid)
{
  int i=0,j=0;
  int is,ie,js,je;
  int nx1,nx2;
  int iprob;
  Real x1c,x2c;
  Real x1f,x2f;
  Real x1size,x2size;
  Real rad,amp,vflow,diag;
  Real **az;

  is = pGrid->is; ie = pGrid->ie;
  js = pGrid->js; je = pGrid->je;
  nx1 = (ie-is)+1 + 2*nghost;
  nx2 = (je-js)+1 + 2*nghost;

  if (((je-js) == 0)) {
    ath_error("[field_loop]: This problem can only be run in 2D or 3D\n");
  }

  if ((az = (Real**)calloc_2d_array(/*1, */nx2, nx1, sizeof(Real))) == NULL) {
    ath_error("[field_loop]: Error allocating memory for vector pot\n");
  }

/* Read initial conditions */

  rad = par_getd("problem","rad");
  amp = par_getd("problem","amp");
  vflow = par_getd("problem","vflow");
  iprob = par_getd("problem","iprob");

/* Use vector potential to initialize field loop */

  for (j=js; j<=je+1; j++) {
  for (i=is; i<=ie+1; i++) {
    cc_pos(pGrid,i,j,&x1c,&x2c);
    x1f = x1c - 0.5*pGrid->dx1;
    x2f = x2c - 0.5*pGrid->dx2;
     
/* (iprob=1): field loop in x1-x2 plane (cylinder in 3D) */

    if(iprob==1) {  
      if ((x1f*x1f + x2f*x2f) < rad*rad) {
        az[j][i] = amp*(rad - sqrt(x1f*x1f + x2f*x2f));
      }
    }

  }}

  x1size = pGrid->dx1*(Real)par_geti("grid","Nx1");
  x2size = pGrid->dx2*(Real)par_geti("grid","Nx2");

  diag = sqrt(x1size*x1size + x2size*x2size);
  for (j=js; j<=je; j++) {
  for (i=is; i<=ie; i++) {
     pGrid->U[j][i].d = 1.0;
     pGrid->U[j][i].M1 = pGrid->U[j][i].d*vflow*x1size/diag;
     pGrid->U[j][i].M2 = pGrid->U[j][i].d*vflow*x2size/diag;
     pGrid->U[j][i].M3 = 0.0;
     pGrid->B1i[j][i] = (az[j+1][i] - az[j][i])/pGrid->dx2;
     pGrid->B2i[j][i] = -(az[j][i+1] - az[j][i])/pGrid->dx1;
     pGrid->B3i[j][i] = 0.0; 
  }}

/* boundary conditions on interface B */

  i = ie+1;
    for (j=js; j<=je; j++) {
      pGrid->B1i[j][i] = (az[j+1][i] - az[j][i])/pGrid->dx2;
    }

  j = je+1;
    for (i=is; i<=ie; i++) {
      pGrid->B2i[j][i] = -(az[j][i+1] - az[j][i])/pGrid->dx1;
    }

/* initialize total energy and cell-centered B */

    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pGrid->U[j][i].B1c = 0.5*(pGrid->B1i[j][i  ] +
				      pGrid->B1i[j][i+1]);
        pGrid->U[j][i].B2c = 0.5*(pGrid->B2i[j  ][i] +
				      pGrid->B2i[j+1][i]);
	pGrid->U[j][i].B3c = pGrid->B3i[j][i];

        pGrid->U[j][i].E = 1.0/Gamma_1
	  + 0.5*(SQR(pGrid->U[j][i].B1c) + SQR(pGrid->U[j][i].B2c)
          + SQR(pGrid->U[j][i].B3c))
	  + 0.5*(SQR(pGrid->U[j][i].M1) + SQR(pGrid->U[j][i].M2)
	+ SQR(pGrid->U[j][i].M3))/pGrid->U[j][i].d;
      }
    }

  free_2d_array((void**)az);

}

/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 * current() - computes x3-component of current
 * Bp2()     - computes magnetic pressure (Bx2 + By2)
 * color()   - returns first passively advected scalar s[0]
 *----------------------------------------------------------------------------*/

void problem_write_restart(Grid *pG, FILE *fp)
{
  return;
}

void problem_read_restart(Grid *pG, FILE *fp)
{
  return;
}

static Real current(const Grid *pG, const int i, const int j)
{
  return ((pG->B2i[j][i]-pG->B2i[j][i-1])/pG->dx1 - 
	  (pG->B1i[j][i]-pG->B1i[j-1][i])/pG->dx2);
}

static Real Bp2(const Grid *pG, const int i, const int j)
{
  return (pG->U[j][i].B1c*pG->U[j][i].B1c + 
	  pG->U[j][i].B2c*pG->U[j][i].B2c);
}


Gasfun_t get_usr_expr(const char *expr)
{
  if(strcmp(expr,"J3")==0) return current;
  else if(strcmp(expr,"Bp2")==0) return Bp2;
  return NULL;
}

void Userwork_in_loop(Grid *pGrid)
{
}

void Userwork_after_loop(Grid *pGrid)
{
}

#endif /* FIELD_LOOP */
