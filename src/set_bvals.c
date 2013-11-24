/* DONE - OK */


// #include "copyright.h"
/*==============================================================================
 * FILE: set_bvals.c
 *
 * PURPOSE: Sets boundary conditions (quantities in ghost zones) on each edge
 *   of a Grid.  Each edge of a Grid represents either:
 *    (1) the physical boundary of computational Domain; in which case BCs are
 *        specified by an integer flag input by user (or by user-defined BC
 *        function in the problem file)
 *    (2) the boundary between Grids resulting from decomposition of a larger
 *        computational domain using MPI; in which case BCs are handled
 *        by MPI calls
 *    (3) an internal boundary between fine/coarse grid levels in a nested grid;
 *        in which case BCs require prolongation and restriction operators (and
 *        possibly some combination of (1) and (2) as well!)
 *   This file contains functions called in the main loop that can handle each
 *   of these cases.  The naming convention for BCs is:
 *       ibc_x1 = Inner Boundary Condition for x1
 *       obc_x1 = Outer Boundary Condition for x1
 *   similarly for ibc_x2; obc_x2; ibc_x3; obc_x3
 *
 * For case (1) -- PHYSICAL BOUNDARIES
 *   The values of the integer flags are:
 *       1 = reflecting; 2 = outflow; 4 = periodic
 *   Following ZEUS conventions, 3 would be flow-in (ghost zones held at
 *   pre-determined fixed values), however in Athena instead we use pointers to
 *   user-defined BC functions for flow-in.
 *
 * For case (2) -- MPI BOUNDARIES
 *   We do the parallel synchronization by having every grid:
 *     1) Pack and send data to the grid on right  [send_ox1()]
 *     2) Listen to the left, unpack and set data  [receive_ix1()]
 *     3) Pack and send data to the grid on left   [send_ix1()]
 *     4) Listen to the right, unpack and set data [receive_ox1()]
 *   If the grid is at the edge of the Domain, we set BCs as in case (1) or (3).
 *   Currently the code uses NON-BLOCKING sends (MPI_Isend) and BLOCKING
 *   receives (MPI_Recv).  Some optimization could be achieved by interleaving
 *   non-blocking sends (MPI_Isend) and computations.
 *
 * For case (3) -- INTERNAL GRID LEVEL BOUNDARIES
 *
 * The type of BC is unchanged throughout a calculation.  Thus, during setup
 * we determine the BC type, and set a pointer to the appropriate BC function
 * using set_bvals_init().  MPI calls are used if the grid ID number to the
 * left or right is >= 0.
 * 
 * With SELF-GRAVITY: BCs for Phi have to be set independently of the Gas.  The
 *   argument list of the BC functions contain an integer flag which determines
 *   whether BCs for Phi (flag=1) or the Gas structure (flag=0) are set.
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   set_bvals()      - calls appropriate functions to set ghost cells
 *   set_bvals_init() - sets function pointers used by set_bvals() based on flag
 *   set_bvals_fun()  - enrolls a pointer to a user-defined BC function
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "prototypes.h"

/* boundary condition function pointers */
static VBCFun_t apply_ix1 = NULL, apply_ox1 = NULL;
static VBCFun_t apply_ix2 = NULL, apply_ox2 = NULL;
static VBCFun_t apply_ix3 = NULL, apply_ox3 = NULL;

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 *   reflect_???()  - apply reflecting BCs at boundary ???
 *   outflow_???()  - apply outflow BCs at boundary ???
 *   periodic_???() - apply periodic BCs at boundary ???
 *   send_???()     - MPI send of data at ??? boundary
 *   receive_???()  - MPI receive of data at ??? boundary
 *============================================================================*/

static void reflect_ix1(Grid *pG, int var_flag);
static void reflect_ox1(Grid *pG, int var_flag);
static void reflect_ix2(Grid *pG, int var_flag);
static void reflect_ox2(Grid *pG, int var_flag);
static void reflect_ix3(Grid *pG, int var_flag);
static void reflect_ox3(Grid *pG, int var_flag);

static void outflow_ix1(Grid *pG, int var_flag);
static void outflow_ox1(Grid *pG, int var_flag);
static void outflow_ix2(Grid *pG, int var_flag);
static void outflow_ox2(Grid *pG, int var_flag);
static void outflow_ix3(Grid *pG, int var_flag);
static void outflow_ox3(Grid *pG, int var_flag);

static void periodic_ix1(Grid *pG, int var_flag);
static void periodic_ox1(Grid *pG, int var_flag);
static void periodic_ix2(Grid *pG, int var_flag);
static void periodic_ox2(Grid *pG, int var_flag);
static void periodic_ix3(Grid *pG, int var_flag);
static void periodic_ox3(Grid *pG, int var_flag);


/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/
/* set_bvals:  calls appropriate functions to set ghost zones.  The function
 *   pointers (*apply_???) are set during initialization by set_bvals_init()
 *   to be either a user-defined function, or one of the functions corresponding
 *   to reflecting, periodic, or outflow.  If the left- or right-Grid ID numbers
 *   are >= 1 (neighboring grids exist), then MPI calls are used.
 *
 * Order for updating boundary conditions must always be x1-x2-x3 in order to
 * fill the corner cells properly
 */

void set_bvals(Grid *pGrid, int var_flag)
{

/*--- Step 1. ------------------------------------------------------------------
 * Boundary Conditions in x1-direction */

  if (pGrid->Nx1 > 1){

/* Physical boundaries on both left and right */
    //if (pGrid->rx1_id < 0 && pGrid->lx1_id < 0) {
      (*apply_ix1)(pGrid, var_flag);
      (*apply_ox1)(pGrid, var_flag);
    //}

  }

/*--- Step 2. ------------------------------------------------------------------
 * Boundary Conditions in x2-direction */

  if (pGrid->Nx2 > 1){


/* Physical boundaries on both left and right */
    //if (pGrid->rx2_id < 0 && pGrid->lx2_id < 0) {
      (*apply_ix2)(pGrid, var_flag);
      (*apply_ox2)(pGrid, var_flag);
    //}

  }

  return;
}

/*----------------------------------------------------------------------------*/
/* set_bvals_init:  sets function pointers for physical boundaries during
 *   initialization, allocates memory for send/receive buffers with MPI
 */

void set_bvals_init(Grid *pG/*, Domain *pD*/)
{
  int ibc_x1, obc_x1; /* x1 inner and outer boundary condition flag */
  int ibc_x2, obc_x2; /* x2 inner and outer boundary condition flag */
  // int ibc_x3, obc_x3; /* x3 inner and outer boundary condition flag */

/* Set function pointers for physical boundaries in x1-direction */

  if(pG->Nx1 > 1) {
    if(apply_ix1 == NULL){

      ibc_x1 = par_geti("grid","ibc_x1");
      switch(ibc_x1){

      case 1: /* Reflecting */
	apply_ix1 = reflect_ix1;
	break;

      case 2: /* Outflow */
	apply_ix1 = outflow_ix1;
	break;

      case 4: /* Periodic */
	apply_ix1 = periodic_ix1;
	break;

      default:
	ath_perr(-1,"[set_bvals_init]: ibc_x1 = %d unknown\n",ibc_x1);
	exit(EXIT_FAILURE);
      }

    }

    if(apply_ox1 == NULL){

      obc_x1 = par_geti("grid","obc_x1");
      switch(obc_x1){

      case 1: /* Reflecting */
	apply_ox1 = reflect_ox1;
	break;

      case 2: /* Outflow */
	apply_ox1 = outflow_ox1;
	break;

      case 4: /* Periodic */
	apply_ox1 = periodic_ox1;
	break;

      default:
	ath_perr(-1,"[set_bvals_init]: obc_x1 = %d unknown\n",obc_x1);
	exit(EXIT_FAILURE);
      }

    }
  }

/* Set function pointers for physical boundaries in x2-direction */

  if(pG->Nx2 > 1) {
    if(apply_ix2 == NULL){

      ibc_x2 = par_geti("grid","ibc_x2");
      switch(ibc_x2){

      case 1: /* Reflecting */
	apply_ix2 = reflect_ix2;
	break;

      case 2: /* Outflow */
	apply_ix2 = outflow_ix2;
	break;

      case 4: /* Periodic */
	apply_ix2 = periodic_ix2;
	break;

      default:
	ath_perr(-1,"[set_bvals_init]: ibc_x2 = %d unknown\n",ibc_x2);
	exit(EXIT_FAILURE);
      }

    }

    if(apply_ox2 == NULL){

      obc_x2 = par_geti("grid","obc_x2");
      switch(obc_x2){

      case 1: /* Reflecting */
	apply_ox2 = reflect_ox2;
	break;

      case 2: /* Outflow */
	apply_ox2 = outflow_ox2;
	break;

      case 4: /* Periodic */
	apply_ox2 = periodic_ox2;
	break;

      default:
	ath_perr(-1,"[set_bvals_init]: obc_x2 = %d unknown\n",obc_x2);
	exit(EXIT_FAILURE);
      }

    }
  }

  return;
}

/*----------------------------------------------------------------------------*/
/* set_bvals_fun:  sets function pointers for user-defined BCs in problem file
 */

void set_bvals_fun(enum Direction dir, VBCFun_t prob_bc)
{
  switch(dir){
  case left_x1:
    apply_ix1 = prob_bc;
    break;
  case right_x1:
    apply_ox1 = prob_bc;
    break;
  case left_x2:
    apply_ix2 = prob_bc;
    break;
  case right_x2:
    apply_ox2 = prob_bc;
    break;
  case left_x3:
    apply_ix3 = prob_bc;
    break;
  case right_x3:
    apply_ox3 = prob_bc;
    break;
  default:
    ath_perr(-1,"[set_bvals_fun]: Unknown direction = %d\n",dir);
    exit(EXIT_FAILURE);
  }
  return;
}

/*=========================== PRIVATE FUNCTIONS ==============================*/
/* Following are the functions:
 *   reflecting_???
 *   outflow_???
 *   periodic_???
 *   send_???
 *   receive_???
 * where ???=[ix1,ox1,ix2,ox2,ix3,ox3]
 */

/*----------------------------------------------------------------------------*/
/* REFLECTING boundary conditions, Inner x1 boundary (ibc_x1=1)
 * Phi set by multipole expansion external to this function
 */

static void reflect_ix1(Grid *pGrid, int var_flag)
{
  int is = pGrid->is;
  int js = pGrid->js, je = pGrid->je;
//  int ks = pGrid->ks, ke = pGrid->ke;
  int i,j; //,k;
  int ju, ku; /* j-upper, k-upper */

  if (var_flag == 1) return;

 // for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->U[j][is-i]    =  pGrid->U[j][is+(i-1)];
        pGrid->U[j][is-i].M1 = -pGrid->U[j][is-i].M1; /* reflect 1-mom. */
      }
    }
  //}

//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B1i[j][is-i] = pGrid->B1i[j][is+(i-1)];
      }
    }
//  }

  if (pGrid->Nx2 > 1) ju=je+1; else ju=je;
  //for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[j][is-i] = pGrid->B2i[j][is+(i-1)];
      }
    }
  //}


 // if (pGrid->Nx3 > 1) ku=ke+1; else
  //  ku=ke;
  //for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B3i[j][is-i] = pGrid->B3i[j][is+(i-1)];
      }
    }
  //}


  return;
}

/*----------------------------------------------------------------------------*/
/* REFLECTING boundary conditions, Outer x1 boundary (obc_x1=1)
 * Phi set by multipole expansion external to this function
 */

static void reflect_ox1(Grid *pGrid, int var_flag)
{
  int ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
//  int ks = pGrid->ks, ke = pGrid->ke;
  int i,j; //,k;
  int ju, ku; /* j-upper, k-upper */

  if (var_flag == 1) return;

  // for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->U[j][ie+i] = pGrid->U[j][ie-(i-1)];
        pGrid->U[j][ie+i].M1 = -pGrid->U[j][ie+i].M1; /* reflect 1-mom. */
      }
    }
  // }

/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=2; i<=nghost; i++) {
        pGrid->B1i[j][ie+i] = pGrid->B1i[j][ie-(i-2)];
      }
    }
 // }

  if (pGrid->Nx2 > 1) ju=je+1; else ju=je;
  //for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[j][ie+i] = pGrid->B2i[j][ie-(i-1)];
      }
    }
  //}

//  if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
 // for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B3i[j][ie+i] = pGrid->B3i[j][ie-(i-1)];
      }
    }
 // }
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* REFLECTING boundary conditions, Inner x2 boundary (ibc_x2=1)
 * Phi set by multipole expansion external to this function
 */

static void reflect_ix2(Grid *pGrid, int var_flag)
{
  int js = pGrid->js;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,/* k,*/ il,iu; /* i-lower/upper */
// #ifdef MHD
//   int ku; /* k-upper */
// #endif

  if (var_flag == 1) return;

  if (pGrid->Nx1 > 1){
    iu = pGrid->ie + nghost;
    il = pGrid->is - nghost;
  } else {
    iu = pGrid->ie;
    il = pGrid->is;
  }

  //for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->U[js-j][i]    =  pGrid->U[js+(j-1)][i];
        pGrid->U[js-j][i].M2 = -pGrid->U[js-j][i].M2; /* reflect 2-mom. */
      }
    }
  //}

// #ifdef MHD
//  for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B1i[js-j][i] = pGrid->B1i[js+(j-1)][i];
      }
    }
//  }

//  for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B2i[js-j][i] = pGrid->B2i[js+(j-1)][i];
      }
    }
//  }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  // for (k=ks; k<=ku; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B3i[js-j][i] = pGrid->B3i[js+(j-1)][i];
      }
    }
  // }
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* REFLECTING boundary conditions, Outer x2 boundary (obc_x2=1)
 * Phi set by multipole expansion external to this function
 */

static void reflect_ox2(Grid *pGrid, int var_flag)
{
  int je = pGrid->je;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,/* k,*/ il,iu; /* i-lower/upper */
/*
#ifdef MHD
  int ku; /* k-upper */
// #endif
// */

  if (var_flag == 1) return;

  if (pGrid->Nx1 > 1){
    iu = pGrid->ie + nghost;
    il = pGrid->is - nghost;
  } else {
    iu = pGrid->ie;
    il = pGrid->is;
  }

  // for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->U[je+j][i]    =  pGrid->U[je-(j-1)][i];
        pGrid->U[je+j][i].M2 = -pGrid->U[je+j][i].M2; /* reflect 2-mom. */
      }
    }
  // }

// #ifdef MHD
  // for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B1i[je+j][i] = pGrid->B1i[je-(j-1)][i];
      }
    }
  // }

/* Note that j=je+1 is not a boundary condition for the interface field B2i */
  // for (k=ks; k<=ke; k++) {
    for (j=2; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B2i[je+j][i] = pGrid->B2i[je-(j-2)][i];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  // for (k=ks; k<=ku; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B3i[je+j][i] = pGrid->B3i[je-(j-1)][i];
      }
    }
  // }
// #endif /* MHD */

  return;
}


/*----------------------------------------------------------------------------*/
/* OUTFLOW boundary conditionss, Inner x1 boundary (ibc_x1=2)
 * Phi set by multipole expansion external to this function
 */

static void outflow_ix1(Grid *pGrid, int var_flag)
{
  int is = pGrid->is;
  int js = pGrid->js, je = pGrid->je;
//   int ks = pGrid->ks, ke = pGrid->ke;
  int i,j; //,k;
// #ifdef MHD
  int ju; //, ku; /* j-upper, k-upper */
// #endif

  if (var_flag == 1) return;

 // for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->U[j][is-i] = pGrid->U[j][is];
      }
    }
  // }

// #ifdef MHD
//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B1i[j][is-i] = pGrid->B1i[j][is];
      }
    }
//   }

  if (pGrid->Nx2 > 1) ju=je+1; else ju=je;
  // for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[j][is-i] = pGrid->B2i[j][is];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  // for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B3i[j][is-i] = pGrid->B3i[j][is];
      }
    }
  // }
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* OUTFLOW boundary conditions, Outer x1 boundary (obc_x1=2)
 * Phi set by multipole expansion external to this function
 */

static void outflow_ox1(Grid *pGrid, int var_flag)
{
  int ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j; //,k;
// #ifdef MHD
  int ju; //, ku; /* j-upper, k-upper */
// #endif

  if (var_flag == 1) return;

 //  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->U[j][ie+i] = pGrid->U[j][ie];
      }
    }
  // }

// #ifdef MHD
/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=2; i<=nghost; i++) {
        pGrid->B1i[j][ie+i] = pGrid->B1i[j][ie];
      }
    }
//  }

  if (pGrid->Nx2 > 1) ju=je+1; else ju=je;
 //  for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[j][ie+i] = pGrid->B2i[j][ie];
      }
    }
 //  }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  // for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B3i[j][ie+i] = pGrid->B3i[j][ie];
      }
    }
  // }
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* OUTFLOW boundary conditions, Inner x2 boundary (ibc_x2=2)
 * Phi set by multipole expansion external to this function
 */

static void outflow_ix2(Grid *pGrid, int var_flag)
{
  int js = pGrid->js;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,/* k,*/ il,iu; /* i-lower/upper */
// #ifdef MHD
//  int ku; /* k-upper */
//#endif

  if (var_flag == 1) return;

  if (pGrid->Nx1 > 1){
    iu = pGrid->ie + nghost;
    il = pGrid->is - nghost;
  } else {
    iu = pGrid->ie;
    il = pGrid->is;
  }

//   for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->U[js-j][i] = pGrid->U[js][i];
      }
    }
  // }

// #ifdef MHD
  // for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B1i[js-j][i] = pGrid->B1i[js][i];
      }
    }
 // }

  // for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B2i[js-j][i] = pGrid->B2i[js][i];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  // for (k=ks; k<=ku; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B3i[js-j][i] = pGrid->B3i[js][i];
      }
    }
  //}
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* OUTFLOW boundary conditions, Outer x2 boundary (obc_x2=2)
 * Phi set by multipole expansion external to this function
 */

static void outflow_ox2(Grid *pGrid, int var_flag)
{
  int je = pGrid->je;
  //int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,/*k,*/il,iu; /* i-lower/upper */
//#ifdef MHD
//  int ku; /* k-upper */
//#endif

  if (var_flag == 1) return;

  if (pGrid->Nx1 > 1){
    iu = pGrid->ie + nghost;
    il = pGrid->is - nghost;
  } else {
    iu = pGrid->ie;
    il = pGrid->is;
  }

/* Note that j=je+1 is not a boundary condition for the interface field B2i */

//  for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->U[je+j][i] = pGrid->U[je][i];
      }
    }
//  }

//#ifdef MHD
  //for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B1i[je+j][i] = pGrid->B1i[je][i];
      }
    }
  //}

/* Note that j=je+1 is not a boundary condition for the interface field B2i */
  // for (k=ks; k<=ke; k++) {
    for (j=2; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B2i[je+j][i] = pGrid->B2i[je][i];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  //for (k=ks; k<=ku; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B3i[je+j][i] = pGrid->B3i[je][i];
      }
    }
  // }
//#endif /* MHD */

  return;
}


/*----------------------------------------------------------------------------*/
/* PERIODIC boundary conditions, Inner x1 boundary (ibc_x1=4)
 */

static void periodic_ix1(Grid *pGrid, int var_flag)
{
  int is = pGrid->is, ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
//  int ks = pGrid->ks, ke = pGrid->ke;
  int i,j; //,k;
// #ifdef MHD
  int ju; //, ku; /* j-upper, k-upper */
// #endif

  if (var_flag == 1) {
    return;
  }

  //for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->U[j][is-i] = pGrid->U[j][ie-(i-1)];
      }
    }
  //}

//#ifdef MHD
//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B1i[j][is-i] = pGrid->B1i[j][ie-(i-1)];
      }
    }
 // }

  if (pGrid->Nx2 > 1) ju=je+1; else ju=je;
  // for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[j][is-i] = pGrid->B2i[j][ie-(i-1)];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  // for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B3i[j][is-i] = pGrid->B3i[j][ie-(i-1)];
      }
    }
  //}
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* PERIODIC boundary conditions (cont), Outer x1 boundary (obc_x1=4)
 */

static void periodic_ox1(Grid *pGrid, int var_flag)
{
  int is = pGrid->is, ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j; //,k;
// #ifdef MHD
  int ju; //, ku; /* j-upper, k-upper */
// #endif

  if (var_flag == 1) {
    return;
  }

//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->U[j][ie+i] = pGrid->U[j][is+(i-1)];
      }
    }
//  }

// #ifdef MHD
/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
//  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=2; i<=nghost; i++) {
        pGrid->B1i[j][ie+i] = pGrid->B1i[j][is+(i-1)];
      }
    }
//  }

  if (pGrid->Nx2 > 1) ju=je+1; else ju=je;
  // for (k=ks; k<=ke; k++) {
    for (j=js; j<=ju; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B2i[j][ie+i] = pGrid->B2i[j][is+(i-1)];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
  //for (k=ks; k<=ku; k++) {
    for (j=js; j<=je; j++) {
      for (i=1; i<=nghost; i++) {
        pGrid->B3i[j][ie+i] = pGrid->B3i[j][is+(i-1)];
      }
    }
  // }
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* PERIODIC boundary conditions (cont), Inner x2 boundary (ibc_x2=4)
 */

static void periodic_ix2(Grid *pGrid, int var_flag)
{
  int js = pGrid->js, je = pGrid->je;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,/*k,*/il,iu; /* i-lower/upper */
// #ifdef MHD
//  int ku; /* k-upper */
//#endif

  if (pGrid->Nx1 > 1){
    iu = pGrid->ie + nghost;
    il = pGrid->is - nghost;
  } else {
    iu = pGrid->ie;
    il = pGrid->is;
  }

  if (var_flag == 1) {
    return;
  }

//  for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->U[js-j][i] = pGrid->U[je-(j-1)][i];
      }
    }
 // }

// #ifdef MHD
//  for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B1i[js-j][i] = pGrid->B1i[je-(j-1)][i];
      }
    }
 // }

  // for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B2i[js-j][i] = pGrid->B2i[je-(j-1)][i];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
 //  for (k=ks; k<=ku; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B3i[js-j][i] = pGrid->B3i[je-(j-1)][i];
      }
    }
  // }
// #endif /* MHD */

  return;
}

/*----------------------------------------------------------------------------*/
/* PERIODIC boundary conditions (cont), Outer x2 boundary (obc_x2=4)
 */

static void periodic_ox2(Grid *pGrid, int var_flag)
{
  int js = pGrid->js, je = pGrid->je;
  // int ks = pGrid->ks, ke = pGrid->ke;
  int i,j,/* k,*/il,iu; /* i-lower/upper */
/*
#ifdef MHD
  int ku; /* k-upper */
///#endif
//*/

  if (pGrid->Nx1 > 1){
    iu = pGrid->ie + nghost;
    il = pGrid->is - nghost;
  } else {
    iu = pGrid->ie;
    il = pGrid->is;
  }

  if (var_flag == 1) {
    return;
  }

//  for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->U[je+j][i] = pGrid->U[js+(j-1)][i];
      }
    }
 // }

// #ifdef MHD
  // for (k=ks; k<=ke; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B1i[je+j][i] = pGrid->B1i[js+(j-1)][i];
      }
    }
 // }

/* Note that j=je+1 is not a boundary condition for the interface field B2i */
  // for (k=ks; k<=ke; k++) {
    for (j=2; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B2i[je+j][i] = pGrid->B2i[js+(j-1)][i];
      }
    }
  // }

  // if (pGrid->Nx3 > 1) ku=ke+1; else ku=ke;
//  for (k=ks; k<=ku; k++) {
    for (j=1; j<=nghost; j++) {
      for (i=il; i<=iu; i++) {
        pGrid->B3i[je+j][i] = pGrid->B3i[js+(j-1)][i];
      }
    }
//  }
// #endif /* MHD */

  return;
}

