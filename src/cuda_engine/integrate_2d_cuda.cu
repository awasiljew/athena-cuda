/*
 * integrate_2d_cuda.c
 *
 *  Created on: 2010-02-25
 *      Author: adam
 */

//Needed headers
#include <stdio.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "../debug_tools_cuda.h"
#include "../debug_tools_cuda.c"

int ESYS_ROE_CUDA_FLAG = 0;

//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#define CUDA_KERNEL_DIM(...)
#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>
#endif



/************** Device variables ***************/
/* maximum wavespeed used by H-correction, value passed from integrator */
__device__ __const__ Real etah_dev = 0.0;

/* The L/R states of conserved variables and fluxes at each cell face */
__device__ Cons1D *Ul_x1Face_dev=NULL, *Ur_x1Face_dev=NULL;
__device__ Cons1D *Ul_x2Face_dev=NULL, *Ur_x2Face_dev=NULL;
__device__ Cons1D *x1Flux_dev=NULL, *x2Flux_dev=NULL;

/* The interface magnetic fields and emfs */
__device__ Real *B1_x1Face_dev=NULL, *B2_x2Face_dev=NULL;
__device__ Real *emf3_dev=NULL, *emf3_cc_dev=NULL;

/* 1D scratch vectors used by lr_states and flux functions */
__device__ Prim1D *W_dev=NULL, *Wl_dev=NULL, *Wr_dev=NULL;
__device__ Real *Bxc_dev=NULL, *Bxi_dev=NULL;
__device__ Cons1D *U1d_dev=NULL;

/* density at t^{n+1/2} needed by both MHD and to make gravity 2nd order */
__device__ Real *dhalf_dev = NULL;

/* Work pointer */
__device__ Real** pW_dev;

/*********************** Inclusions from other kernel functions ***********************/
#include "kernels/cc_pos.cu"
#include "kernels/esys_roe_adb_mhd.cu"
#include "kernels/flux_hlle.cu"
#include "kernels/flux_roe.cu"
#include "kernels/esys_prim_adb_mhd.cu"
#include "kernels/lr_states.cu"
#include "kernels/var_conversions.cu"
#include "kernels/slices1D.cu"
#include "kernels/mhd_source_terms.cu"
#include "kernels/emf.cu"
#include "kernels/update_emf.cu"
#include "kernels/flux_transverse_correction.cu"
#include "kernels/dhalf.cu"
#include "kernels/slices1D_2.cu"
#include "kernels/update_cc_fluxes.cu"


void showError(const char * cmd, cudaError_t code) {
	if(cmd != NULL) {
		if(code != cudaSuccess) {
			fprintf(stderr,"%s %s\n", cmd, cudaGetErrorString(code));
		}
	} else {
		if(code != cudaSuccess) {
			fprintf(stderr,"%s\n", cudaGetErrorString(code));
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////                     INTEGRATE 2D CUDA FUNCTION              /////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C"
void integrate_2d_cuda(Grid_gpu *pG_gpu) {

	int is = pG_gpu->is, ie = pG_gpu->ie;
	int js = pG_gpu->js, je = pG_gpu->je;
	int i,il,iu;
	int j,jl,ju;
	int sizex = pG_gpu->Nx1+2*nghost;
	int sizey = pG_gpu->Nx2+2*nghost;

	Real hdt = 0.5*pG_gpu->dt;
	Real dtodx1 = pG_gpu->dt/pG_gpu->dx1;
	Real hdtodx1 = 0.5*dtodx1;
	Real dtodx2 = pG_gpu->dt/pG_gpu->dx2;
	Real hdtodx2 = 0.5*dtodx2;

	il = is - 2;
	iu = ie + 2;

	jl = js - 2;
	ju = je + 2;

	cudaError_t code;

	int nnBlocks = (sizex*sizey)/(BLOCK_SIZE) + ((sizex*sizey) % (BLOCK_SIZE) ? 1 : 0);
	int nBlocks = sizex/BLOCK_SIZE + (sizex % BLOCK_SIZE ? 1 : 0);
	int nBlocks_y = sizey/BLOCK_SIZE + (sizey % BLOCK_SIZE ? 1 : 0);

	/* Step 1a */
	load_1a_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (U1d_dev, Bxc_dev, pG_gpu->U, is-nghost, ie+nghost, jl, ju, sizex);

	code = cudaMemcpy(B1_x1Face_dev, pG_gpu->B1i, sizeof(Real)*sizex*sizey, cudaMemcpyDeviceToDevice);
	showError("Copy B1_x1Face: ", code);

	/* Step 1b */
	for (j=jl; j<=ju; j++) {
		Cons1D_to_Prim1D_1b_dev CUDA_KERNEL_DIM(nBlocks, BLOCK_SIZE) (U1d_dev, W_dev, Bxc_dev, is-nghost, ie+nghost, j, sizex, Gamma_1);
		lr_states_cu_1b_dev CUDA_KERNEL_DIM(nBlocks, BLOCK_SIZE) (W_dev, Bxc_dev, pG_gpu->dt, dtodx1, is-2, ie+2, j, sizex, Wl_dev, Wr_dev, Gamma);
		addMHDSourceTerms_dev_half CUDA_KERNEL_DIM(nBlocks, BLOCK_SIZE) (Wl_dev, Wr_dev, pG_gpu->U, pG_gpu->B1i, pG_gpu->dx1, is-1, iu, j, sizex, hdt);
	}

	/* Step 1e */
	Cons1D_to_Prim1D_Slice1D_1e CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x1Face_dev, Ur_x1Face_dev, Wl_dev, Wr_dev, pG_gpu->B1i, x1Flux_dev, is-1, iu, jl, ju, sizex, Gamma_1, Gamma_2);

	/* Step 2a */
	load_2a_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (U1d_dev, Bxc_dev, pG_gpu->U, il, iu, js-nghost, je+nghost, sizex, sizey);

	code = cudaMemcpy(B2_x2Face_dev, pG_gpu->B2i, sizeof(Real)*sizex*sizey, cudaMemcpyDeviceToDevice);
	showError("Copy B2_x2Face: ", code);

	/* Step 2b */
	for (i=il; i<=iu; i++) {
		Cons1D_to_Prim1D_2b_dev CUDA_KERNEL_DIM(nBlocks_y, BLOCK_SIZE) (U1d_dev, W_dev, Bxc_dev, js-nghost, je+nghost, i, sizex, sizey, Gamma_1);
		lr_states_cu_1b_dev CUDA_KERNEL_DIM(nBlocks_y, BLOCK_SIZE) (W_dev, Bxc_dev, pG_gpu->dt, dtodx2, js-2, je+2, i, sizex, Wl_dev, Wr_dev, Gamma);
		addMHDSourceTerms_dev_half_y CUDA_KERNEL_DIM(nBlocks_y, BLOCK_SIZE) (Wl_dev, Wr_dev, pG_gpu->U, pG_gpu->B2i, pG_gpu->dx2, js-1, ju, i, sizex, sizey, hdt);
	}

	/* Step 2e */
	Cons1D_to_Prim1D_Slice1D_2e CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x2Face_dev, Ur_x2Face_dev, Wl_dev, Wr_dev, pG_gpu->B2i, x2Flux_dev, il, iu, js-1, ju, sizex, sizex, Gamma_1, Gamma_2);

	/* Step 3*/
	emf_3_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (emf3_cc_dev, pG_gpu->U, is-2, ie+2, js-2, je+2, sizex);

	/* Step 4 */
	integrate_emf3_corner_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (x1Flux_dev, x2Flux_dev, emf3_dev, emf3_cc_dev, is-1, ie+2, js-1, je+2, sizex);
	updateMagneticField_4a_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (B1_x1Face_dev, B2_x2Face_dev, emf3_dev, is-1, ie+1, js-1, je+1, sizex, hdtodx2, hdtodx1);
	updateMagneticField_4b_dev CUDA_KERNEL_DIM(nBlocks_y, BLOCK_SIZE) (B1_x1Face_dev, emf3_dev, js-1, je+1, ie+2, sizex, hdtodx2);
	updateMagneticField_4c_dev CUDA_KERNEL_DIM(nBlocks, BLOCK_SIZE) (B2_x2Face_dev, emf3_dev, is-1, ie+1, je+2, sizex, hdtodx1);

	/* Step 5 */
	correctTransverseFluxGradients_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x1Face_dev, Ur_x1Face_dev, x2Flux_dev, is-1, ie+2, js-1, je+1, sizex, hdtodx2, hdtodx1);
	addMHDSourceTerms_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x1Face_dev, Ur_x1Face_dev, pG_gpu->U, pG_gpu->B1i, is-1, ie+2, js-1, je+1, sizex, hdtodx2, hdtodx1);

	/* Step 6 */
	correctTransverseFluxGradients_A_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x2Face_dev, Ur_x2Face_dev, x1Flux_dev, is-1, ie+1, js-1, je+2, sizex, hdtodx2, hdtodx1);
	addMHDSourceTerms_B_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x2Face_dev, Ur_x2Face_dev, pG_gpu->U, pG_gpu->B2i, is-1, ie+1, js-1, je + 2, sizex, hdtodx2, hdtodx1);

	/* Step 7 */
	if (dhalf_dev != NULL){
		dhalf_init_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (dhalf_dev, x1Flux_dev, x2Flux_dev, pG_gpu->U, is-1, ie+1, js-1, je+1, sizex, hdtodx1, hdtodx2);
	}
	cc_emf3_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (dhalf_dev, x1Flux_dev, x2Flux_dev, B1_x1Face_dev, B2_x2Face_dev, emf3_cc_dev, pG_gpu, pG_gpu->U, is-1, ie+1, js-1, je+1, sizex, hdtodx1, hdtodx2);

	/* Step 8 */
	Cons1D_to_Prim1D_Slice1D_8b CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x1Face_dev, Ur_x1Face_dev, Wl_dev, Wr_dev, B1_x1Face_dev, x1Flux_dev, is, ie+1, js-1, je+1, sizex, Gamma_1, Gamma_2);
	Cons1D_to_Prim1D_Slice1D_8c CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (Ul_x2Face_dev, Ur_x2Face_dev, Wl_dev, Wr_dev, B2_x2Face_dev, x2Flux_dev, is-1, ie+1, js, je+1, sizex, Gamma_1, Gamma_2);

	/* Step 9 */
	integrate_emf3_corner_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (x1Flux_dev, x2Flux_dev, emf3_dev, emf3_cc_dev, is-1, ie+2, js-1, je+2, sizex);
	updateMagneticField_9a_dev CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (emf3_dev, pG_gpu->B1i, pG_gpu->B2i, is, ie, js, je, sizex, dtodx2, dtodx1);
	updateMagneticField_9b_dev CUDA_KERNEL_DIM(nBlocks_y, BLOCK_SIZE) (emf3_dev, pG_gpu->B1i, js, je, ie, sizex, dtodx2);
	updateMagneticField_9c_dev CUDA_KERNEL_DIM(nBlocks, BLOCK_SIZE) (emf3_dev, pG_gpu->B2i, is, ie, je, sizex, dtodx1);

	/* Step 11 */
	update_cc_x1_Flux CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (x1Flux_dev, pG_gpu->U, is, ie, js, je, sizex, dtodx1);
	update_cc_x2_Flux CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (x2Flux_dev, pG_gpu->U, is, ie, js, je, sizex, dtodx2);

	/* Step 13 */
	update_cc_mf CUDA_KERNEL_DIM(nnBlocks, BLOCK_SIZE) (pG_gpu->U, pG_gpu->B1i, pG_gpu->B2i, pG_gpu->B3i, is, ie, js, je, sizex);

	code = cudaThreadSynchronize();
	showError("Synchronize: ", code);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C"
void checkAllocationError(cudaError_t status) {
    if (status != cudaSuccess){
        fprintf(stderr, "%s", cudaGetErrorString(status));
    }
}

extern "C"
void checkDeallocationError(cudaError_t status) {
    if (status != cudaSuccess){
    	fprintf(stderr, "%s", cudaGetErrorString(status));
    }
}

extern "C"
void integrate_init_2d_cu(int nx1, int nx2)
{
  int nmax;
  int Nx1 = nx1 + 2*nghost;
  int Nx2 = nx2 + 2*nghost;
  nmax = MAX(Nx1,Nx2);

  cudaError_t status = cudaMalloc((void**)&emf3_dev, sizeof(Real)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&emf3_cc_dev, sizeof(Real)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Bxc_dev, sizeof(Real)*nmax*nmax);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Bxi_dev, sizeof(Real)*nmax*nmax);
  checkAllocationError(status);

  status = cudaMalloc((void**)&B1_x1Face_dev, sizeof(Real)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&B2_x2Face_dev, sizeof(Real)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&U1d_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&W_dev, sizeof(Prim1D)*nmax*nmax);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Wl_dev, sizeof(Prim1D)*nmax*nmax);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Wr_dev, sizeof(Prim1D)*nmax*nmax);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Ul_x1Face_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Ur_x1Face_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Ul_x2Face_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&Ur_x2Face_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&x1Flux_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&x2Flux_dev, sizeof(Cons1D)*Nx2*Nx1);
  checkAllocationError(status);

  status = cudaMalloc((void**)&dhalf_dev, sizeof(Real)*Nx2*Nx1);
  checkAllocationError(status);

}

extern "C"
void integrate_destruct_2d_cu() {
  cudaError_t status = cudaFree(emf3_dev);
  checkDeallocationError(status);

  status = cudaFree(emf3_cc_dev);
  checkDeallocationError(status);

  status = cudaFree(Bxc_dev);
  checkDeallocationError(status);

  status = cudaFree(Bxi_dev);
  checkDeallocationError(status);

  status = cudaFree(B1_x1Face_dev);
  checkDeallocationError(status);

  status = cudaFree(B2_x2Face_dev);
  checkDeallocationError(status);

  status = cudaFree(U1d_dev);
  checkDeallocationError(status);

  status = cudaFree(W_dev);
  checkDeallocationError(status);

  status = cudaFree(Wl_dev);
  checkDeallocationError(status);

  status = cudaFree(Wr_dev);
  checkDeallocationError(status);

  status = cudaFree(Ul_x1Face_dev);
  checkDeallocationError(status);

  status = cudaFree(Ur_x1Face_dev);
  checkDeallocationError(status);

  status = cudaFree(Ul_x2Face_dev);
  checkDeallocationError(status);

  status = cudaFree(Ur_x2Face_dev);
  checkDeallocationError(status);

  status = cudaFree(x1Flux_dev);
  checkDeallocationError(status);

  status = cudaFree(x2Flux_dev);
  checkDeallocationError(status);

  status = cudaFree(dhalf_dev);
  checkDeallocationError(status);
}

extern "C"
void lr_states_init_cu(int nx1, int nx2)
{
  int /*i, */nmax;
  nmax =  nx1 > nx2  ? nx1 : nx2;
  nmax += 2*nghost;

  cudaError_t status = cudaMalloc((void**)&pW_dev, sizeof(Real*)*nmax);
  checkAllocationError(status);
}

void** calloc_2d_array(size_t nr, size_t nc, size_t size)
{
  void **array;
  size_t i;

  if((array = (void **)calloc(nr,sizeof(void*))) == NULL){
    return NULL;
  }

  if((array[0] = (void *)calloc(nr*nc,size)) == NULL){
    free((void *)array);
    return NULL;
  }

  for(i=1; i<nr; i++){
    array[i] = (void *)((unsigned char *)array[0] + i*nc*size);
  }

  return array;
}

void free_2d_array(void *array)
{
  void **ta = (void **)array;

  free(ta[0]);
  free(array);
}
