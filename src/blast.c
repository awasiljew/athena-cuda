#include "copyright.h"
/*==============================================================================
 * FILE: blast.c
 *
 * PURPOSE: Problem generator for spherical blast wave problem.
 *
 * REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
 *   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include "debug_tools_cuda.h"

/*----------------------------------------------------------------------------*/
/* problem:  */

#ifdef BLAST_PROBLEM

void problem(Grid *pGrid)
{
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  Real pressure,prat,rad,da,pa,ua,va,wa,x1,x2,x3;
  Real bxa,bya,bza,b0=0.0;
  Real rin;
  double theta;

  rin = par_getd("problem","radius");
  pa  = par_getd("problem","pamb");
  prat = par_getd("problem","prat");
  b0 = par_getd("problem","b0");
  theta = (PI/180.0)*par_getd("problem","angle");

/* setup uniform ambient medium with spherical over-pressured region */

  da = 1.0;
  ua = 0.0;
  va = 0.0;
  wa = 0.0;
  bxa = b0*cos(theta);
  bya = b0*sin(theta);
  bza = 0.0;
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
	pGrid->U[j][i].d  = da;
	pGrid->U[j][i].M1 = da*ua;
	pGrid->U[j][i].M2 = da*va;
	pGrid->U[j][i].M3 = da*wa;
	pGrid->B1i[j][i] = bxa;
	pGrid->B2i[j][i] = bya;
	pGrid->B3i[j][i] = bza;
	pGrid->U[j][i].B1c = bxa;
	pGrid->U[j][i].B2c = bya;
	pGrid->U[j][i].B3c = bza;
	if (i == ie && ie > is) pGrid->B1i[j][i+1] = bxa;
	if (j == je && je > js) pGrid->B2i[j+1][i] = bya;
	cc_pos(pGrid,i,j,&x1,&x2);
	rad = sqrt(x1*x1 + x2*x2);
	pressure = pa;
	if (rad < rin) pressure = prat*pa;
        pGrid->U[j][i].E = pressure/Gamma_1
          + 0.5*(bxa*bxa + bya*bya + bza*bza)
          + 0.5*da*(ua*ua + va*va + wa*wa);
      }
    }
}

/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 *----------------------------------------------------------------------------*/

void problem_write_restart(Grid *pG, FILE *fp)
{
  return;
}

void problem_read_restart(Grid *pG, FILE *fp)
{
  return;
}

Gasfun_t get_usr_expr(const char *expr)
{
  return NULL;
}

void Userwork_in_loop(Grid *pGrid)
{
}

void Userwork_after_loop(Grid *pGrid)
{
}

#endif /* BLAST_PROBLEM */
