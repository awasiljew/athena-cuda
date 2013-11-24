/**
 * Modified by awasiljew -> awasiljew@gmail.com
 * 22.11.2009
 * We assuming MHD, adiabatic equation of state, no H_CORRECTION, NSCALARS =0, NO_SELF_GRAVITY, SELF_GRAVITY_NONE,
 * GET_FLUXES=flux_roe - Roe Flux, StaticGravPot = NULL
 */

#ifndef ATHENA_H
#define ATHENA_H 
/*==============================================================================
 * FILE: athena.h
 *
 * PURPOSE: Contains definitions of the following data types and structures:
 *   Real  - either float or double, depending on configure option
 *   Gas   - cell-centered conserved variables
 *   Prim  - cell-centered primitive variables
 *   Cons1D - conserved variables in 1D: same as Gas minus Bx
 *   Prim1D - primitive variables in 1D: same as Prim minus Bx
 *   Grid   - everything needed by a Grid: arrays of Gas, B, indices, time, etc.
 *   Domain - Indices and IDs of each Grid block across all processors
 *   Output - everything associated with an individual output: time, type, etc.
 *============================================================================*/
#include "defs.h"

/*----------------------------------------------------------------------------*/
/* variable type Real
 *   depends on macro set by configure
 */ 

#if defined(SINGLE_PREC)
typedef float  Real;
#elif defined(DOUBLE_PREC)
typedef double Real;
#else
# error "Not a valid precision flag"
#endif

/*----------------------------------------------------------------------------*/
/* structure Gas: conserved variables 
 *  IMPORTANT!! The order of the elements in Gas CANNOT be changed.
 */

typedef struct Gas_s{
  Real d;			/* density */
  Real M1;			/* Momenta in 1,2,3.  Use 1,2,3 to label */
  Real M2;                      /* directions in anticipation of         */
  Real M3;                      /* covariant coordinates in future       */
  Real E;			/* Total energy density */
  Real B1c;			/* cell centered magnetic fields in 1,2,3 */
  Real B2c;
  Real B3c;
}Gas;

/*----------------------------------------------------------------------------*/
/* structure Prim: primitive variables 
 *  IMPORTANT!! The order of the elements in Prim CANNOT be changed.
 */

typedef struct Prim_s{
  Real d;			/* density  */
  Real V1;			/* Velocity in 1,2,3 */
  Real V2;
  Real V3;
  Real P;			/* pressure */
  Real B1c;                     /* cell centered magnetic fields in 1,2,3 */
  Real B2c;
  Real B3c;
}Prim;

/*----------------------------------------------------------------------------*/
/* structure Cons1D:  conserved variables in 1D (does not contain Bx)
 *  IMPORTANT!! The order of the elements in Cons1D CANNOT be changed.
 */

typedef struct Cons1D_s{
  Real d;			/* density */
  Real Mx;			/* Momenta in X,Y,Z.  Use X,Y,Z now instead  */
  Real My;                      /* of 1,2,3 since this structure can contain */
  Real Mz;                      /* a slice in any dimension: 1,2,or 3        */
  Real E;			/* Total energy density */
  Real By;			/* cell centered magnetic fields in Y,Z */
  Real Bz;
}Cons1D;

/*----------------------------------------------------------------------------*/
/* structure Prim1D:  primitive variables in 1D (does not contain Bx)
 *  IMPORTANT!! The order of the elements in Prim1D CANNOT be changed.
 */

typedef struct Prim1D_s{
  Real d;			/* density */
  Real Vx;			/* Velocity in X,Y,Z */
  Real Vy;
  Real Vz;
  Real P;			/* pressure */
  Real By;			/* cell centered magnetic fields in Y,Z */
  Real Bz;
}Prim1D;

/*----------------------------------------------------------------------------*/
/* structure Grid: All data needed by a single processor to integrate equations
 *   Initialized by init_grid().  By using an array of Gas, rather than arrays
 *   of each variable, we guarantee data for each cell are contiguous in memory.
 */

typedef struct Grid_s{
  Gas **U;			/* pointer to a 2D array of Gas'es */
  Real **B1i,**B2i, **B3i;          /* pointer to a 2D array of interface B's */
  Real x1_0;		        /* x1-position of coordinate ix = 0 */
  Real x2_0;		        /* x2-position of coordinate jx = 0 */
  Real dx1,dx2; 	       	/* cell size */
  Real dt,time;			/* time step, absolute time */
  int nstep;			/* number of integration steps taken */
  int Nx1,Nx2;   		/* number of zones in x1, x2, x3 direction */
  int is,ie;			/* start/end cell index in x1 direction */
  int js,je;			/* start/end cell index in x2 direction */
  int idisp;                    /* coordinate ix = index i + idisp */
  int jdisp;                    /* coordinate jx = index j + jdisp */
  char *outfilename;		/* basename for output files */
}Grid;

typedef struct Grid_s_gpu {
  Gas *U;                      /* pointer to a 2D array of Gas'es */
  Real *B1i,*B2i, *B3i;        /* pointer to a 2D array of interface B's */
  Real x1_0;                   /* x1-position of coordinate ix = 0 */
  Real x2_0;                   /* x2-position of coordinate jx = 0 */
  Real dx1,dx2;                /* cell size */
  Real dt,time;                /* time step, absolute time */
  int nstep;                   /* number of integration steps taken */
  int Nx1,Nx2;                 /* number of zones in x1, x2, x3 direction */
  int is,ie;                   /* start/end cell index in x1 direction */
  int js,je;                   /* start/end cell index in x2 direction */
  int idisp;                   /* coordinate ix = index i + idisp */
  int jdisp;                   /* coordinate jx = index j + jdisp */
  char *outfilename;           /* basename for output files */
} Grid_gpu;

/*----------------------------------------------------------------------------*/
/* structure Output: everything for outputs */
  
struct Output_s;
typedef void (*VGFunout_t)(Grid *pGrid, /*Domain *pD, */struct Output_s *pout);
typedef Real (*Gasfun_t)(const Grid *pG, const int i, const int j/*, const int k*/);

typedef struct Output_s{
  int n;          /* the N from the <outputN> block of this output */
  Real dt;        /* time interval between outputs  */
  Real t;         /* next time to output */
  int num;        /* dump number (0=first) */
  char *out;      /* variable (or user fun) to be output */
  char *id;       /* filename is of the form <basename>[.idump][.id].<ext> */

/* variables which describe data min/max */
  Real dmin;      /* user defined min for scaling data */
  Real dmax;      /* user defined max for scaling data */
  Real gmin;      /* computed global min (over all data output so far) */
  Real gmax;      /* computed global max (over all data output so far) */
  int sdmin;      /* 0 = auto scale, otherwise use dmin */
  int sdmax;      /* 0 = auto scale, otherwise use dmax */

/* variables which describe coordinates of output data volume */
  int ndim;       /* 3=cube 2=slice 1=vector 0=scalar */
  int ix1l, ix1u; /* lower/upper x1 indices for data slice  -1 = all data */
  int ix2l, ix2u; /* lower/upper x2 indices for data slice  -1 = all data */
  int ix3l, ix3u; /* lower/upper x3 indices for data slice  -1 = all data */
  int Nx1;        /* number of grid points to be output in x1 */
  int Nx2;        /* number of grid points to be output in x2 */
  int Nx3;        /* number of grid points to be output in x3 */
  Real x1_0, dx1; /* origin and grid spacing of output slice in x1 */
  Real x2_0, dx2; /* origin and grid spacing of output slice in x2 */
  Real x3_0, dx3; /* origin and grid spacing of output slice in x3 */

/* variables which describe output format */
  char *out_fmt;  /* output format = {bin, tab, hdf, hst, pgm, ppm, ...} */
  char *dat_fmt;  /* format string for tabular type output, e.g. "%10.5e" */
  char *palette;  /* name of palette for RGB conversions */
  float *rgb;     /* array of RGB[256*3] values derived from palette */
  float *der;     /* helper array of derivatives for interpolation into RGB */

/* pointers to output functions; data expressions */
  VGFunout_t fun; /* function pointer */
  Gasfun_t expr;  /* expression for output */
} Output;


/* prototype for functions that compute static gravitational
 * potential.  These functions are set in problem generator, and used by
 * integrators */
typedef Real (*GravPotFun_t)(const Real x1, const Real x2, const Real x3);

/* Directions for the set_bvals_fun() function */
enum Direction {left_x1, right_x1, left_x2, right_x2, left_x3, right_x3};

/* Definitions of various functions */
typedef void (*VBCFun_t)(Grid *pG, int var_flag);    /* void boundary cond fn */
typedef void (*VGFun_t) (Grid *pG);                     /* void grid function */
typedef void (*VGDFun_t)(Grid *pG);     /*void grid + domain func */

#endif /* ATHENA_H */
