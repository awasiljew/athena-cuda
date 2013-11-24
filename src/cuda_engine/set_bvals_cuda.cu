#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "../prototypes.h"
#include <cuda.h>
#include <cuda_runtime.h>

//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

/* Only for 2D */
static int ibc_x1, obc_x1; /* x1 inner and outer boundary condition flag */
static int ibc_x2, obc_x2; /* x2 inner and outer boundary condition flag */

///////////////////// Reflect ////////////////////////////


///////////// ix1 /////////////////////////////

__global__ void reflect_ix1_cu_1step(Gas *U, Real *B1i, int js, int je, int is,
		int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		U[j * sizex + is - i].d = U[j * sizex + is + (i - 1)].d;
		U[j * sizex + is - i].B1c = U[j * sizex + is + (i - 1)].B1c;
		U[j * sizex + is - i].B2c = U[j * sizex + is + (i - 1)].B2c;
		U[j * sizex + is - i].B3c = U[j * sizex + is + (i - 1)].B3c;
		U[j * sizex + is - i].E = U[j * sizex + is + (i - 1)].E;
		U[j * sizex + is - i].M2 = U[j * sizex + is + (i - 1)].M2;
		U[j * sizex + is - i].M3 = U[j * sizex + is + (i - 1)].M3;

		U[j * sizex + is - i].M1 = -U[j * sizex + is - i].M1; /* reflect 1-mom. */
		B1i[j * sizex + is - i] = B1i[j * sizex + is + (i - 1)];
	}
}

__global__ void reflect_ix1_cu_2step(Real* B2i, int js, int ju, int is,
		int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > ju)
		return;

	for (int i = 1; i <= nghost; i++) {
		B2i[j * sizex + is - i] = B2i[j * sizex + is + (i - 1)];
	}
}

__global__ void reflect_ix1_cu_3step(Real* B3i, int js, int je, int is,
		int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		B3i[j * sizex + is - i] = B3i[j * sizex + is + (i - 1)];
	}
}


void reflect_ix1_cu(Grid_gpu *pGrid)
{
	int is = pGrid->is;
	int js = pGrid->js, je = pGrid->je, ju;
	int sizex = pGrid->Nx1+2*nghost;
	int nBlocks = sizex/BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	reflect_ix1_cu_1step<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, js, je, is, sizex);

	if (pGrid->Nx2 > 1)
		ju = je + 1;
	else
		ju = je;

	reflect_ix1_cu_2step<<<nBlocks, BLOCK_SIZE>>>(pGrid->B2i, js, ju, is, sizex);

	reflect_ix1_cu_3step<<<nBlocks, BLOCK_SIZE>>>(pGrid->B3i, js, je, is, sizex);
}

///////////// ox1 /////////////////////////////

__global__ void reflect_ox1_cu_step1(Gas* U, int js, int je, int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		U[j * sizex + ie + i] = U[j * sizex + ie - (i - 1)];
		U[j * sizex + ie + i].M1 = -U[j * sizex + ie + i].M1; /* reflect 1-mom. */
	}
}

__global__ void reflect_ox1_cu_step2(Real* B1i, int js, int je, int ie,
		int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
	for (int i = 2; i <= nghost; i++) {
		B1i[j * sizex + ie + i] = B1i[j * sizex + ie - (i - 2)];
	}
}

__global__ void reflect_ox1_cu_step3(Real* B2i, int js, int ju, int ie,
		int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > ju)
		return;

	/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
	for (int i = 1; i <= nghost; i++) {
		B2i[j * sizex + ie + i] = B2i[j * sizex + ie - (i - 1)];
	}
}

__global__ void reflect_ox1_cu_step4(Real* B3i, int js, int je, int ie,
		int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
	for (int i = 1; i <= nghost; i++) {
		B3i[j * sizex + ie + i] = B3i[j * sizex + ie - (i - 1)];
	}
}


void reflect_ox1_cu(Grid_gpu *pG)
{
	  int ie = pG->ie;
	  int js = pG->js, je = pG->je;
	  int ju;
	  int sizex = pG->Nx1+2*nghost;
	  int nBlocks = sizex/BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	  reflect_ox1_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pG->U, js, je, ie, sizex);

	  reflect_ox1_cu_step2<<<nBlocks, BLOCK_SIZE>>>(pG->B1i, js, je, ie, sizex);

	  if (pG->Nx2 > 1) ju=je+1; else ju=je;

	  reflect_ox1_cu_step3<<<nBlocks, BLOCK_SIZE>>>(pG->B2i, js, ju, ie, sizex);

	  reflect_ox1_cu_step4<<<nBlocks, BLOCK_SIZE>>>(pG->B3i, js, je, ie, sizex);

}

////////////////////////////////// ix2 /////////////////////////////////////////

__global__ void reflect_ix2_cu_step1(Gas* U, Real *B1i, Real* B2i, Real* B3i,
		int il, int iu, int js, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j = 1; j <= nghost; j++) {
		U[(js - j) * sizex + i].d = U[(js + (j - 1)) * sizex + i].d;
		U[(js - j) * sizex + i].B1c = U[(js + (j - 1)) * sizex + i].B1c;
		U[(js - j) * sizex + i].B2c = U[(js + (j - 1)) * sizex + i].B2c;
		U[(js - j) * sizex + i].B3c = U[(js + (j - 1)) * sizex + i].B3c;
		U[(js - j) * sizex + i].E = U[(js + (j - 1)) * sizex + i].E;
		U[(js - j) * sizex + i].M1 = U[(js + (j - 1)) * sizex + i].M1;
		U[(js - j) * sizex + i].M3 = U[(js + (j - 1)) * sizex + i].M3;

		U[(js - j) * sizex + i].M2 = -U[(js - j) * sizex + i].M2; /* reflect 2-mom. */

		B1i[(js - j) * sizex + i] = B1i[(js + (j - 1)) * sizex + i];

		B2i[(js - j) * sizex + i] = B2i[(js + (j - 1)) * sizex + i];

		B3i[(js - j) * sizex + i] = B3i[(js + (j - 1)) * sizex + i];

	}
}

void reflect_ix2_cu(Grid_gpu *pGrid) {
	int js = pGrid->js;
	int il, iu;
	int sizex = pGrid->Nx1 + 2 * nghost;
	int nBlocks = sizex / BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	if (pGrid->Nx1 > 1) {
		iu = pGrid->ie + nghost;
		il = pGrid->is - nghost;
	} else {
		iu = pGrid->ie;
		il = pGrid->is;
	}

	reflect_ix2_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, pGrid->B2i, pGrid->B3i, il, iu, js, sizex);

}

////////////////////////// ox2 /////////////////////////////////////

__global__ void reflect_ox2_cu_step1(Gas *U, Real* B1i, int il, int iu, int je, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j=1; j<=nghost; j++) {
        U[(je+j)*sizex+i].d =  U[(je-(j-1))*sizex+i].d;
        U[(je+j)*sizex+i].M1 =  U[(je-(j-1))*sizex+i].M1;
        U[(je+j)*sizex+i].M3 =  U[(je-(j-1))*sizex+i].M3;
	    U[(je+j)*sizex+i].M2 = -U[(je+j)*sizex+i].M2; /* reflect 2-mom. */
	    U[(je+j)*sizex+i].E =  U[(je-(j-1))*sizex+i].E;
	    U[(je+j)*sizex+i].B1c =  U[(je-(j-1))*sizex+i].B1c;
	    U[(je+j)*sizex+i].B2c =  U[(je-(j-1))*sizex+i].B2c;
	    U[(je+j)*sizex+i].B3c =  U[(je-(j-1))*sizex+i].B3c;

	    B1i[(je+j)*sizex+i] = B1i[(je-(j-1))*sizex+i];
	}
}

__global__ void reflect_ox2_cu_step2(Real* B2i, int il, int iu, int je, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j=2; j<=nghost; j++) {
	    B2i[(je+j)*sizex+i] = B2i[(je-(j-2))*sizex+i];
	}
}

__global__ void reflect_ox2_cu_step3(Real* B3i, int il, int iu, int je, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j=1; j<=nghost; j++) {
		B3i[(je+j)*sizex+i] = B3i[(je-(j-1))*sizex+i];
	}
}


void reflect_ox2_cu(Grid_gpu *pGrid)
{
	int je = pGrid->je;
	int il,iu;
	int sizex = pGrid->Nx1 + 2 * nghost;
	int nBlocks = sizex / BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	  if (pGrid->Nx1 > 1){
	    iu = pGrid->ie + nghost;
	    il = pGrid->is - nghost;
	  } else {
	    iu = pGrid->ie;
	    il = pGrid->is;
	  }

	  reflect_ox2_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, il, iu, je, sizex);

	/* Note that j=je+1 is not a boundary condition for the interface field B2i */
	  reflect_ox2_cu_step2<<<nBlocks, BLOCK_SIZE>>>(pGrid->B2i, il, iu, je, sizex);

	  reflect_ox2_cu_step3<<<nBlocks, BLOCK_SIZE>>>(pGrid->B3i, il, iu, je, sizex);
}

////////////////////// Outflow /////////////////////////////
void outflow_ix1_cu(Grid_gpu *pG)
{

}

void outflow_ox1_cu(Grid_gpu *pG)
{

}

void outflow_ix2_cu(Grid_gpu *pG)
{

}

void outflow_ox2_cu(Grid_gpu *pG)
{

}

/////////////////////// Periodic ////////////////////////////


////////////////////////////// ix1 //////////////////////////////////////////
__global__ void periodic_ix1_cu_step1(Gas *U, Real *B1i, int js, int je, int is, int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		U[j*sizex+(is - i)].d = U[j*sizex+(ie - (i - 1))].d;
		U[j*sizex+(is - i)].M1 = U[j*sizex+(ie - (i - 1))].M1;
		U[j*sizex+(is - i)].M2 = U[j*sizex+(ie - (i - 1))].M2;
		U[j*sizex+(is - i)].M3 = U[j*sizex+(ie - (i - 1))].M3;
		U[j*sizex+(is - i)].E = U[j*sizex+(ie - (i - 1))].E;
		U[j*sizex+(is - i)].B1c = U[j*sizex+(ie - (i - 1))].B1c;
		U[j*sizex+(is - i)].B2c = U[j*sizex+(ie - (i - 1))].B2c;
		U[j*sizex+(is - i)].B3c = U[j*sizex+(ie - (i - 1))].B3c;

		B1i[j*sizex+(is-i)] = B1i[j*sizex+(ie-(i-1))];

	}
}

__global__ void periodic_ix1_cu_step2(Real *B2i, int js, int ju, int is, int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > ju)
		return;

	for (int i = 1; i <= nghost; i++) {
		B2i[j*sizex+(is-i)] = B2i[j*sizex+(ie-(i-1))];
	}
}

__global__ void periodic_ix1_cu_step3(Real *B3i, int js, int je, int is, int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		B3i[j*sizex+(is-i)] = B3i[j*sizex+(ie-(i-1))];
	}
}


void periodic_ix1_cu(Grid_gpu *pGrid) {
	int is = pGrid->is, ie = pGrid->ie;
	int js = pGrid->js, je = pGrid->je;
	int ju;
	int sizex = pGrid->Nx1 + 2 * nghost;
	int nBlocks = sizex / BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	periodic_ix1_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, js, je, is, ie, sizex);

	if (pGrid->Nx2 > 1)
		ju = je + 1;
	else
		ju = je;

	periodic_ix1_cu_step2<<<nBlocks, BLOCK_SIZE>>>(pGrid->B2i, js, ju, is, ie, sizex);

	periodic_ix1_cu_step3<<<nBlocks, BLOCK_SIZE>>>(pGrid->B3i, js, je, is, ie, sizex);

}

/////////////////////////////////////////// ox1 /////////////////////////////////////////
__global__ void periodic_ox1_cu_step1(Gas *U, int js, int je, int is, int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		U[j * sizex + (ie + i)].d = U[j * sizex + (is + (i - 1))].d;
		U[j * sizex + (ie + i)].E = U[j * sizex + (is + (i - 1))].E;
		U[j * sizex + (ie + i)].M1 = U[j * sizex + (is + (i - 1))].M1;
		U[j * sizex + (ie + i)].M2 = U[j * sizex + (is + (i - 1))].M2;
		U[j * sizex + (ie + i)].M3 = U[j * sizex + (is + (i - 1))].M3;
		U[j * sizex + (ie + i)].B1c = U[j * sizex + (is + (i - 1))].B1c;
		U[j * sizex + (ie + i)].B2c = U[j * sizex + (is + (i - 1))].B2c;
		U[j * sizex + (ie + i)].B3c = U[j * sizex + (is + (i - 1))].B3c;

	}
}

__global__ void periodic_ox1_cu_step2(Real *B1i, int js, int je, int is,
		int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 2; i <= nghost; i++) {
		B1i[j * sizex + (ie + i)] = B1i[j * sizex + (is + (i - 1))];
	}
}

__global__ void periodic_ox1_cu_step3(Real *B2i, int js, int ju, int is,
		int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > ju)
		return;

	for (int i = 1; i <= nghost; i++) {
		B2i[j * sizex + (ie + i)] = B2i[j * sizex + (is + (i - 1))];
	}
}

__global__ void periodic_ox1_cu_step4(Real *B3i, int js, int je, int is,
		int ie, int sizex) {
	/* Calculate and check index */
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < js || j > je)
		return;

	for (int i = 1; i <= nghost; i++) {
		B3i[j * sizex + (ie + i)] = B3i[j * sizex + (is + (i - 1))];
	}
}


void periodic_ox1_cu(Grid_gpu *pGrid) {
	int is = pGrid->is, ie = pGrid->ie;
	int js = pGrid->js, je = pGrid->je;
	int ju;
	int sizex = pGrid->Nx1 + 2 * nghost;
	int nBlocks = sizex / BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	periodic_ox1_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, js, je, is, ie, sizex);

	/* Note that i=ie+1 is not a boundary condition for the interface field B1i */
	periodic_ox1_cu_step2<<<nBlocks, BLOCK_SIZE>>>(pGrid->B1i, js, je, is, ie, sizex);

	if (pGrid->Nx2 > 1)
		ju = je + 1;
	else
		ju = je;

	periodic_ox1_cu_step3<<<nBlocks, BLOCK_SIZE>>>(pGrid->B2i, js, ju, is, ie, sizex);

	periodic_ox1_cu_step4<<<nBlocks, BLOCK_SIZE>>>(pGrid->B3i, js, je, is, ie, sizex);

}

/////////////////////////////////////////// ix2 //////////////////////////////////////////
__global__ void periodic_ix2_cu_step1(Gas *U, Real *B1i, Real *B2i, Real *B3i,
		int il, int iu, int js, int je, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j = 1; j <= nghost; j++) {

		U[(js - j) * sizex + i].d = U[(je - (j - 1)) * sizex + i].d;
		U[(js - j) * sizex + i].E = U[(je - (j - 1)) * sizex + i].E;
		U[(js - j) * sizex + i].M1 = U[(je - (j - 1)) * sizex + i].M1;
		U[(js - j) * sizex + i].M2 = U[(je - (j - 1)) * sizex + i].M2;
		U[(js - j) * sizex + i].M3 = U[(je - (j - 1)) * sizex + i].M3;
		U[(js - j) * sizex + i].B1c = U[(je - (j - 1)) * sizex + i].B1c;
		U[(js - j) * sizex + i].B2c = U[(je - (j - 1)) * sizex + i].B2c;
		U[(js - j) * sizex + i].B3c = U[(je - (j - 1)) * sizex + i].B3c;

		B1i[(js - j) * sizex + i] = B1i[(je - (j - 1)) * sizex + i];

		B2i[(js - j) * sizex + i] = B2i[(je - (j - 1)) * sizex + i];

		B3i[(js - j) * sizex + i] = B3i[(je - (j - 1)) * sizex + i];

	}
}

void periodic_ix2_cu(Grid_gpu *pGrid) {
	int js = pGrid->js, je = pGrid->je;
	int il, iu; /* i-lower/upper */
	int sizex = pGrid->Nx1 + 2 * nghost;
	int nBlocks = sizex / BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	if (pGrid->Nx1 > 1) {
		iu = pGrid->ie + nghost;
		il = pGrid->is - nghost;
	} else {
		iu = pGrid->ie;
		il = pGrid->is;
	}

	periodic_ix2_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, pGrid->B2i, pGrid->B3i, il, iu, js, je, sizex);
}

/////////////////////////////////////////// ox2 //////////////////////////////////////////
__global__ void periodic_ox2_cu_step1(Gas *U, Real *B1i, int il, int iu, int js, int je,
		int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j = 1; j <= nghost; j++) {
		U[(je + j) * sizex + i].d = U[(js + (j - 1)) * sizex + i].d;
		U[(je + j) * sizex + i].E = U[(js + (j - 1)) * sizex + i].E;
		U[(je + j) * sizex + i].M1 = U[(js + (j - 1)) * sizex + i].M1;
		U[(je + j) * sizex + i].M2 = U[(js + (j - 1)) * sizex + i].M2;
		U[(je + j) * sizex + i].M3 = U[(js + (j - 1)) * sizex + i].M3;
		U[(je + j) * sizex + i].B1c = U[(js + (j - 1)) * sizex + i].B1c;
		U[(je + j) * sizex + i].B2c = U[(js + (j - 1)) * sizex + i].B2c;
		U[(je + j) * sizex + i].B3c = U[(js + (j - 1)) * sizex + i].B3c;

		B1i[(je + j) * sizex + i] = B1i[(js + (j - 1)) * sizex + i];
	}
}

__global__ void periodic_ox2_cu_step2(Real *B2i, int il, int iu, int js, int je, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j = 2; j <= nghost; j++) {
		B2i[(je + j) * sizex + i] = B2i[(js + (j - 1)) * sizex + i];
	}
}

__global__ void periodic_ox2_cu_step3(Real *B3i, int il, int iu, int js, int je, int sizex) {
	/* Calculate and check index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < il || i > iu)
		return;

	for (int j = 1; j <= nghost; j++) {
		B3i[(je + j) * sizex + i] = B3i[(js + (j - 1)) * sizex + i];
	}
}

void periodic_ox2_cu(Grid_gpu *pGrid) {
	int js = pGrid->js, je = pGrid->je;
	int il, iu; /* i-lower/upper */
	int sizex = pGrid->Nx1 + 2 * nghost;
	int nBlocks = sizex / BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);

	if (pGrid->Nx1 > 1) {
		iu = pGrid->ie + nghost;
		il = pGrid->is - nghost;
	} else {
		iu = pGrid->ie;
		il = pGrid->is;
	}

	periodic_ox2_cu_step1<<<nBlocks, BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, il, iu, js, je, sizex);

	/* Note that j=je+1 is not a boundary condition for the interface field B2i */
	periodic_ox2_cu_step2<<<nBlocks, BLOCK_SIZE>>>(pGrid->B2i, il, iu, js, je, sizex);

	periodic_ox2_cu_step3<<<nBlocks, BLOCK_SIZE>>>(pGrid->B3i, il, iu, js, je, sizex);
}

///////////////////////////// TODO: outflow ////////////////////////////////////




/*----------------------------------------------------------------------------*/
/* set_bvals_init:  sets function pointers for physical boundaries during
 *   initialization, allocates memory for send/receive buffers with MPI
 */
extern "C"
void set_bvals_init_cu(Grid_gpu *pG, int ix1, int ox1, int ix2, int ox2) {
	/* Read settings */
	if (pG->Nx1 > 1) {
		ibc_x1 = ix1;
		switch (ibc_x1) {
		case 1: /* Reflecting */
			break;
		case 2: /* Outflow */
			break;
		case 4: /* Periodic */
			break;
		default:
			fprintf(stderr, "[set_bvals_init]: ibc_x1 = %d unknown\n", ibc_x1);
			exit(EXIT_FAILURE);
		}

		obc_x1 = ox1;
		switch (obc_x1) {
		case 1: /* Reflecting */
			break;
		case 2: /* Outflow */
			break;
		case 4: /* Periodic */
			break;
		default:
			fprintf(stderr, "[set_bvals_init]: obc_x1 = %d unknown\n", obc_x1);
			exit(EXIT_FAILURE);
		}
	}

	/* Set function pointers for physical boundaries in x2-direction */
	if (pG->Nx2 > 1) {
		ibc_x2 = ix2;
		switch (ibc_x2) {
		case 1: /* Reflecting */
			break;
		case 2: /* Outflow */
			break;
		case 4: /* Periodic */
			break;
		default:
			fprintf(stderr, "[set_bvals_init]: ibc_x2 = %d unknown\n", ibc_x2);
			exit(EXIT_FAILURE);
		}

		obc_x2 = ox2;
		switch (obc_x2) {
		case 1: /* Reflecting */
			break;
		case 2: /* Outflow */
			break;
		case 4: /* Periodic */
			break;
		default:
			fprintf(stderr, "[set_bvals_init]: obc_x2 = %d unknown\n", obc_x2);
			exit(EXIT_FAILURE);
		}
	}
}

extern "C"
void set_bvals_cu(Grid_gpu *pGrid, int var_flag) {
	/* Read settings */
	switch (ibc_x1) {
	case 1: /* Reflecting */
		reflect_ix1_cu(pGrid);
		break;
	case 2: /* Outflow */
		outflow_ix1_cu(pGrid);
		break;
	case 4: /* Periodic */
		periodic_ix1_cu(pGrid);
		break;
	default:
		fprintf(stderr, "[set_bvals_init]: ibc_x1 = %d unknown\n", ibc_x1);
		exit(EXIT_FAILURE);
	}

	switch (obc_x1) {
	case 1: /* Reflecting */
		reflect_ox1_cu(pGrid);
		break;
	case 2: /* Outflow */
		outflow_ox1_cu(pGrid);
		break;
	case 4: /* Periodic */
		periodic_ox1_cu(pGrid);
		break;
	default:
		fprintf(stderr, "[set_bvals_init]: obc_x1 = %d unknown\n", obc_x1);
		exit(EXIT_FAILURE);
	}

	/* Set function pointers for physical boundaries in x2-direction */
	switch (ibc_x2) {
	case 1: /* Reflecting */
		reflect_ix2_cu(pGrid);
		break;
	case 2: /* Outflow */
		outflow_ix2_cu(pGrid);
		break;
	case 4: /* Periodic */
		periodic_ix2_cu(pGrid);
		break;
	default:
		fprintf(stderr, "[set_bvals_init]: ibc_x2 = %d unknown\n", ibc_x2);
		exit(EXIT_FAILURE);
	}

	switch (obc_x2) {
	case 1: /* Reflecting */
		reflect_ox2_cu(pGrid);
		break;
	case 2: /* Outflow */
		outflow_ox2_cu(pGrid);
		break;
	case 4: /* Periodic */
		periodic_ox2_cu(pGrid);
		break;
	default:
		fprintf(stderr, "[set_bvals_init]: obc_x2 = %d unknown\n", obc_x2);
		exit(EXIT_FAILURE);
	}
}

