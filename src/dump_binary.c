/* DONE - OK */

#include "copyright.h"
/*==============================================================================
 * FILE: dump_binary.c
 *
 * PURPOSE: Function to write an unformatted dump of the field variables that
 *   can be read, e.g., by IDL scripts.  Also called by dump_dx().
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   dump_binary -
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

/*----------------------------------------------------------------------------*/
/* dump_binary:  */

void dump_binary(Grid *pGrid,/* Domain *pD, */ Output *pOut)
{
  int dnum = pOut->num;
  FILE *p_binfile;
  char *fname;
  int n,ndata[6];
/* Upper and Lower bounds on i,j for data dump */
  int i, il = pGrid->is, iu = pGrid->ie;
  int j, jl = pGrid->js, ju = pGrid->je;
  float dat[2],*datax,*datay;
  Real *pq,x1,x2;

#ifdef WRITE_GHOST_CELLS
  if(pGrid->Nx1 > 1){
    il = pGrid->is - nghost;
    iu = pGrid->ie + nghost;
  }

  if(pGrid->Nx2 > 1){
    jl = pGrid->js - nghost;
    ju = pGrid->je + nghost;
  }

  if(pGrid->Nx3 > 1){
    kl = pGrid->ks - nghost;
    ku = pGrid->ke + nghost;
  }
#endif /* WRITE_GHOST_CELLS */

  if((fname = fname_construct(pGrid->outfilename,num_digit,dnum,NULL,"bin")) 
     == NULL){
    ath_error("[dump_binary]: Error constructing filename\n");
    return;
  }

  if((p_binfile = fopen(fname,"wb")) == NULL){
    ath_error("[dump_binary]: Unable to open binary dump file\n");
    return;
  }

/* Write number of zones and variables */
  ndata[0] = iu-il+1;
  ndata[1] = ju-jl+1;
  ndata[2] = 0;
  ndata[3] = NVAR;
  ndata[4] = NSCALARS;
  ndata[5] = 0;
  fwrite(ndata,sizeof(int),6,p_binfile);

/* Write (gamma-1) and isothermal sound speed */

  dat[0] = (float)Gamma_1 ;
  dat[1] = (float)0.0;

  fwrite(dat,sizeof(float),2,p_binfile);

/* Write time, dt */

  dat[0] = (float)pGrid->time;
  dat[1] = (float)pGrid->dt;
  fwrite(dat,sizeof(float),2,p_binfile);

/* Allocate Memory */

  if((datax = (float *)malloc(ndata[0]*sizeof(float))) == NULL){
    ath_error("[dump_binary]: malloc failed for temporary array\n");
    return;
  }
  if((datay = (float *)malloc(ndata[1]*sizeof(float))) == NULL){
    ath_error("[dump_binary]: malloc failed for temporary array\n");
    return;
  }


/* compute x,y,z coordinates of cell centers, and write out */

  for (i=il; i<=iu; i++) {
    cc_pos(pGrid,i,jl,&x1,&x2);
    pq = ((Real *) &(x1));
    datax[i-il] = (float)(*pq);
  }
  fwrite(datax,sizeof(float),(size_t)ndata[0],p_binfile);

  for (j=jl; j<=ju; j++) {
    cc_pos(pGrid,il,j,&x1,&x2);
    pq = ((Real *) &(x2));
    datay[j-jl] = (float)(*pq);
  }
  fwrite(datay,sizeof(float),(size_t)ndata[1],p_binfile);


/* Write cell-centered data in Gas array pGrid->U[n] */

  for (n=0;n<NVAR; n++) {
    for (j=0; j<ndata[1]; j++) {
      for (i=0; i<ndata[0]; i++) {
        pq = ((Real *) &(pGrid->U[j+jl][i+il])) + n;
        datax[i] = (float)(*pq);
      }
      fwrite(datax,sizeof(float),(size_t)ndata[0],p_binfile);
    }
  }


/* close file and free memory */
  fclose(p_binfile); 
  free(datax); 
  free(datay); 
  free(fname);
}
