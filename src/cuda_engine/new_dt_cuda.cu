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

__device__ Real *max_dti_array=NULL;

/*----------------------------------------------------------------------------*/
/* new_dt:  */

__global__ void new_dt_1Step_cuda_kernel(Gas *U, Real *B1i, Real* B2i,
		Real* B3i, int is, int ie, int js, int je, int sizex, Real *max_dti_array, Real Gamma,
		Real Gamma_1, Real dx1, Real dx2, int Nx1, int Nx2) {

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = id / sizex;
	int i = id % sizex;

	if (i < is || i > ie || j < js || j > je) {
		max_dti_array[id] = 0.0;
		return;
	} else {
		max_dti_array[id] = 0.0;
	}

	Real di, v1, v2, v3, qsq, p, asq, cf1sq, cf2sq, /*cf3sq,*/ max_dti = 0.0;
	Real b1, b2, b3, bsq, tsum, tdif;

	int ind = j * sizex + i;

	di = 1.0 / (U[ind].d);
	v1 = U[ind].M1 * di;
	v2 = U[ind].M2 * di;
	v3 = U[ind].M3 * di;
	qsq = __dmul_rn(v1, v1) + __dmul_rn(v2, v2) + __dmul_rn(v3, v3);

	/* Use maximum of face-centered fields (always larger than cell-centered B) */
	b1 = U[ind].B1c + fabs((Real) (B1i[ind] - U[ind].B1c));
	b2 = U[ind].B2c + fabs((Real) (B2i[ind] - U[ind].B2c));
	b3 = U[ind].B3c + fabs((Real) (B3i[ind] - U[ind].B3c));
	bsq = __dmul_rn(b1, b1) + __dmul_rn(b2, b2) + __dmul_rn(b3, b3);
	/* compute sound speed squared */
	p = MAX(Gamma_1*(U[ind].E - __dmul_rn(0.5*U[ind].d, qsq) - __dmul_rn(0.5,bsq)), TINY_NUMBER);
	asq = Gamma * p * di;

	/* compute fast magnetosonic speed squared in each direction */
	tsum = __dmul_rn(bsq, di) + asq;
	tdif = __dmul_rn(bsq, di) - asq;
	cf1sq = 0.5 * (tsum + sqrt(__dmul_rn(tdif, tdif) + __dmul_rn(4.0 * asq, (__dmul_rn(b2, b2) + __dmul_rn(b3, b3))	* di)));
	cf2sq = 0.5 * (tsum + sqrt(__dmul_rn(tdif, tdif) + __dmul_rn(4.0 * asq, (__dmul_rn(b1, b1) + __dmul_rn(b3, b3))	* di)));

	/* compute sound speed squared */
	p = MAX(Gamma_1*(U[ind].E - __dmul_rn(0.5*U[ind].d,qsq)), TINY_NUMBER);

	asq = Gamma * p * di;

	/* compute fast magnetosonic speed squared in each direction */
	cf1sq = asq;
	cf2sq = asq;
//	cf3sq = asq;

	/* compute maximum inverse of dt (corresponding to minimum dt) */
	if (Nx1 > 1)
		max_dti = MAX(max_dti,(fabs(v1)+sqrt((Real)cf1sq))/dx1);
	if (Nx2 > 1)
		max_dti = MAX(max_dti,(fabs(v2)+sqrt((Real)cf2sq))/dx2);
	/* Store max_dti to some global memory, then we will found it by reduction like algorithm... */
	max_dti_array[id] = max_dti;

//	for (j=pGrid->js; j<=pGrid->je; j++) {
//	    for (i=pGrid->is; i<=pGrid->ie; i++) {
//	      di = 1.0/(pGrid->U[j][i].d);
//	      v1 = pGrid->U[j][i].M1*di;
//	      v2 = pGrid->U[j][i].M2*di;
//	      v3 = pGrid->U[j][i].M3*di;
//	      qsq = v1*v1 + v2*v2 + v3*v3;
//
//	// #ifdef MHD
//
//	/* Use maximum of face-centered fields (always larger than cell-centered B) */
//	      b1 = pGrid->U[j][i].B1c
//	        + fabs((double)(pGrid->B1i[j][i] - pGrid->U[j][i].B1c));
//	      b2 = pGrid->U[j][i].B2c
//	        + fabs((double)(pGrid->B2i[j][i] - pGrid->U[j][i].B2c));
//	      b3 = pGrid->U[j][i].B3c
//	        + fabs((double)(pGrid->B3i[j][i] - pGrid->U[j][i].B3c));
//	      bsq = b1*b1 + b2*b2 + b3*b3;
//	/* compute sound speed squared */
//	      p = MAX(Gamma_1*(pGrid->U[j][i].E - 0.5*pGrid->U[j][i].d*qsq
//	              - 0.5*bsq), TINY_NUMBER);
//	      asq = Gamma*p*di;
//
//	/* compute fast magnetosonic speed squared in each direction */
//	      tsum = bsq*di + asq;
//	      tdif = bsq*di - asq;
//	      cf1sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b2*b2+b3*b3)*di));
//	      cf2sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b1*b1+b3*b3)*di));
//	      cf3sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b1*b1+b2*b2)*di));
//
//
//	/* compute sound speed squared */
//	// #ifdef ADIABATIC
//	      p = MAX(Gamma_1*(pGrid->U[j][i].E - 0.5*pGrid->U[j][i].d*qsq),
//	              TINY_NUMBER);
//	      asq = Gamma*p*di;
//	// #else
//	//      asq = Iso_csound2;
//	// #endif /* ADIABATIC */
//	/* compute fast magnetosonic speed squared in each direction */
//	      cf1sq = asq;
//	      cf2sq = asq;
//	      cf3sq = asq;
//
//	// #endif /* MHD */
//
//	/* compute maximum inverse of dt (corresponding to minimum dt) */
//	      if (pGrid->Nx1 > 1)
//	        max_dti = MAX(max_dti,(fabs(v1)+sqrt((double)cf1sq))/pGrid->dx1);
//	      if (pGrid->Nx2 > 1)
//	        max_dti = MAX(max_dti,(fabs(v2)+sqrt((double)cf2sq))/pGrid->dx2);
//	   //   if (pGrid->Nx3 > 1)
//	   //     max_dti = MAX(max_dti,(fabs(v3)+sqrt((double)cf3sq))/pGrid->dx3);
//	//
//	  //  }
//	  }}
//
//	/* new timestep.  Limit increase to 2x old value */
//	  if (pGrid->nstep == 0) {
//	    pGrid->dt = CourNo/max_dti;
//	  } else {
//	    pGrid->dt = MIN(2.0*pGrid->dt, CourNo/max_dti);
//	  }

}

__global__ void get_max_dti(int N, int n, Real *max_dti_array) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i1 = (1 << n) * i;
	int i2 = i1 + ( 1 << (n -1 ));
	if(i2 < N) {
		max_dti_array[i1] = MAX(max_dti_array[i1], max_dti_array[i2]);
	}
}

extern "C"
void new_dt_cuda(Grid_gpu *pGrid, Real Gamma, Real Gamma_1, Real CourNo)
{
//	int sizex = pGrid->Nx1+2*nghost;
//	int sizey = pGrid->Nx2+2*nghost;
//	int nnBlocks = (sizex*sizey)/(BLOCK_SIZE) + ((sizex*sizey) % (BLOCK_SIZE) ? 1 : 0);
//
//	new_dt_1Step_cuda_kernel<<<nnBlocks,BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, pGrid->B2i,
//			pGrid->B3i, pGrid->is, pGrid->ie, pGrid->js, pGrid->je, sizex, max_dti_array, Gamma,
//			Gamma_1, pGrid->dx1, pGrid->dx2, pGrid->Nx1, pGrid->Nx2);
//
//	int n = 1;
//	int N = sizex*sizey / (1 << n) + (sizex*sizey) % (1 << n); //How many threads
//	nnBlocks = N/BLOCK_SIZE + (N % BLOCK_SIZE ? 1 : 0);
//
//	while(N > 1) {
//		get_max_dti<<<nnBlocks,BLOCK_SIZE>>>(sizex*sizey, n, max_dti_array);
//		n++;
//		N = sizex*sizey / (1 << n) + ((sizex*sizey) % (1 << n) ? 1 : 0); //How many threads
//		nnBlocks = N/BLOCK_SIZE + (N % BLOCK_SIZE ? 1 : 0);
//	}
//
//	/* Copy max_dti_array[0] as maximum :) */
//	Real max_dti;
//	cudaMemcpy(&max_dti, max_dti_array, sizeof(Real), cudaMemcpyDeviceToHost);
//
//	/* new timestep.  Limit increase to 2x old value */
//	  if (pGrid->nstep == 0) {
//	    pGrid->dt = CourNo/max_dti;
//	  } else {
//	    pGrid->dt = MIN(2.0*pGrid->dt, CourNo/max_dti);
//	  }


	int sizex = pGrid->Nx1+2*nghost;
		int sizey = pGrid->Nx2+2*nghost;
		int nnBlocks = (sizex*sizey)/(BLOCK_SIZE) + ((sizex*sizey) % (BLOCK_SIZE) ? 1 : 0);

		//printf("New dt %d\n", nnBlocks);

		new_dt_1Step_cuda_kernel<<<nnBlocks,BLOCK_SIZE>>>(pGrid->U, pGrid->B1i, pGrid->B2i,
				pGrid->B3i, pGrid->is, pGrid->ie, pGrid->js, pGrid->je, sizex, max_dti_array, Gamma,
				Gamma_1, pGrid->dx1, pGrid->dx2, pGrid->Nx1, pGrid->Nx2);

		//printf("After kernel invoke\n");

		int n = 1;
		int N = sizex*sizey / (1 << n) + (sizex*sizey) % (1 << n); //How many threads
		nnBlocks = N/BLOCK_SIZE + (N % BLOCK_SIZE ? 1 : 0);

		while(N > 1) {
			get_max_dti<<<nnBlocks,BLOCK_SIZE>>>(sizex*sizey, n, max_dti_array);
			n++;
			N = sizex*sizey / (1 << n) + ((sizex*sizey) % (1 << n) ? 1 : 0); //How many threads
			nnBlocks = N/BLOCK_SIZE + (N % BLOCK_SIZE ? 1 : 0);
		}

		/* Copy max_dti_array[0] as maximum :) */
		Real max_dti;
		cudaMemcpy(&max_dti, max_dti_array, sizeof(Real), cudaMemcpyDeviceToHost);

		//printf("After 2nd kernel invoke %e\n", max_dti);

		/* new timestep.  Limit increase to 2x old value */
		  if (pGrid->nstep == 0) {
			//  printf("CourNo %f\n", CourNo);
		    pGrid->dt = CourNo/max_dti;
		  //  printf("Step 0 DT CUDA %e\n", pGrid->dt);
		  } else {
			  //printf("CourNo %f\n", CourNo);
		    pGrid->dt = MIN(2.0*pGrid->dt, CourNo/max_dti);
		  }

		  //pGrid->dt = 0.004; //CHECK WHY DT IS CALCULATING WRONG!!!!!!!!!!!

	//	  printf("DT CUDA %e\n", pGrid->dt);

}

void checkErrorDt(cudaError_t status) {
    if (status != cudaSuccess){
        fprintf(stderr, "%s", cudaGetErrorString(status));
    }
}

extern "C"
void dt_init(Grid_gpu* pGrid) {
	int nx = pGrid->Nx1+2*nghost;
	int ny = pGrid->Nx2+2*nghost;
	printf("Work array for dt %d\n",nx*ny);
	cudaError_t status = cudaMalloc((void**)&max_dti_array, sizeof(Real)*nx*ny);
	checkErrorDt(status);
}

extern "C"
void dt_destruct() {
	cudaError_t status = cudaFree(max_dti_array);
	checkErrorDt(status);
}
