/*==============================================================================
 * FILE: utils_cu.c
 *
 * PURPOSE: A variety of useful utility functions for CUDA.
 *
 * CONTAINS PUBLIC FUNCTIONS:
 *
 * init_grid_gpu()
 * copy_to_gpu_mem()
 * copy_gpu_to_gpu_mem()
 * copy_gpu_mem_to_gpu()
 * copy_to_host_mem()
 *
 *============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "defs.h"
#include "athena.h"
#include "prototypes.h"
#include <cuda.h>
#include <cuda_runtime.h>

/**==============================================================================
 * Initialize grid on device memory (GPU). Initialization is performed through
 * steps:
 *
 * 1. Set simple parameters (single variables) in pG_gpu based on values from pG.
 * 2. Allocate memory (1D arrays) of pG_gpu for gas variables U, B1i, B2i, B3i. Memory is
 *    allocated on device memory, but pointers are stored in host memory (its a CUDA feature).
 * 3. Copy whole structure pG_gpu to pG_gpu_dev. Memory for structure pG_gpu_dev should been
 *    already allocated on device memory. The last step will allow to use all variables
 *    from structure in GPU kernels functions (all parts of structure are inside global device
 *    memory).
 */
//void init_grid_gpu(Grid_gpu *pG_gpu_dev, Grid_gpu *pG_gpu, Grid *pG) {
//
//  cudaError_t code;
//
//  int Nx1T,Nx2T; /* Total Number of grid cells in x1,x2,x3 direction */
//  int ib,jb,kb;
//
//  /* initialize time, nstep */
//  pG_gpu->time = pG->time;
//  pG_gpu->nstep = pG->nstep;
//
//  ib = jb = kb = 0;
//
//  /* ---------------------  Intialize grid in 1-direction --------------------- */
//  /* Initialize is,ie */
//
//  pG_gpu->Nx1 = pG->Nx1;
//  pG_gpu->is = pG->is;
//  pG_gpu->ie = pG->ie;
//  pG_gpu->dx1 = pG->dx1;
//
//  /* Initialize i-displacement, and the x1-position of coordinate ix = 0. */
//
//  pG_gpu->idisp =  pG->idisp = 0;
//  pG_gpu->x1_0 = pG->x1_0;
//
//  /* ---------------------  Intialize grid in 2-direction --------------------- */
//  /* Initialize js,je */
//
//  pG_gpu->Nx2 = pG->Nx2;
//
//  pG_gpu->js = pG->js;
//
//  /* Compute dx2 */
//
//  pG_gpu->dx2 = pG->dx2;
//
//  /* Initialize j-displacement, and the x2-position of coordinate jx = 0. */
//
//  pG_gpu->jdisp = pG->jdisp;
//  pG_gpu->x2_0 = pG->x2_0;
//
//  /* ---------  Allocate 2D arrays to hold Gas based on size of grid --------- */
//
//  if (pG->Nx1 > 1)
//    Nx1T = pG->Nx1 + 2*nghost;
//  else
//    Nx1T = 1;
//
//  if (pG->Nx2 > 1)
//    Nx2T = pG->Nx2 + 2*nghost;
//  else
//    Nx2T = 1;
//
//  /* Allocate memory for gas structures and interface field (device memory) */
//  /* U */
//  code = cudaMalloc((void**)&pG_gpu->U, Nx2T*Nx1T*sizeof(Gas));
//  if(code != cudaSuccess) {
//    ath_error("[cudaMalloc_2d] failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
//        Nx1T, Nx2T, (int)(sizeof(Gas)), cudaGetErrorString(code));
//  }
//  /* B1i */
//  code = cudaMalloc((void**)&pG_gpu->B1i, Nx2T*Nx1T*sizeof(Real));
//  if(code != cudaSuccess) {
//    cudaFree(pG_gpu->U);
//    ath_error("[cudaMalloc_2d] failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
//            Nx1T, Nx2T, (int)(sizeof(Real)), cudaGetErrorString(code));
//  }
//  /* B2i */
//  code = cudaMalloc((void**)&pG_gpu->B2i, Nx2T*Nx1T*sizeof(Real));
//  if(code != cudaSuccess) {
//    cudaFree(pG_gpu->U);
//    cudaFree(pG_gpu->B1i);
//    ath_error("[cudaMalloc_2d] failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
//        Nx1T, Nx2T, (int)(sizeof(Real)), cudaGetErrorString(code));
//  }
//  /* B3i */
//  code = cudaMalloc((void**)&pG_gpu->B3i, Nx2T*Nx1T*sizeof(Real));
//  if(code != cudaSuccess) {
//    cudaFree(pG_gpu->U);
//    cudaFree(pG_gpu->B1i);
//    cudaFree(pG_gpu->B2i);
//    ath_error("[cudaMalloc_2d] failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
//        Nx1T, Nx2T, (int)(sizeof(Real)), cudaGetErrorString(code));
//  }
//
//  /* Copy grid from host to gpu memory */
//  code = cudaMemcpy(pG_gpu_dev, pG_gpu, sizeof(Grid_gpu), cudaMemcpyHostToDevice);
//  if(code != cudaSuccess) {
//    ath_error("Cuda copy grid structure error: %s\n", cudaGetErrorString(code));
//  }
//
//}

void init_grid_gpu(Grid_gpu *pG_gpu, Grid *pG) {

  cudaError_t code;

  int Nx1T,Nx2T; /* Total Number of grid cells in x1,x2,x3 direction */
  //int ib,jb,kb;

  //printf("pG->Nx1 %d pG->Nx2 %d\n", pG->Nx1, pG->Nx2);

  /* initialize time, nstep */
  pG_gpu->time = pG->time;
  pG_gpu->nstep = pG->nstep;

 // ib = jb = kb = 0;

  /* ---------------------  Intialize grid in 1-direction --------------------- */
  /* Initialize is,ie */

  pG_gpu->Nx1 = pG->Nx1;
  pG_gpu->is = pG->is;
  pG_gpu->ie = pG->ie;
  pG_gpu->dx1 = pG->dx1;

  /* Initialize i-displacement, and the x1-position of coordinate ix = 0. */

  pG_gpu->idisp =  pG->idisp = 0;
  pG_gpu->x1_0 = pG->x1_0;

  /* ---------------------  Intialize grid in 2-direction --------------------- */
  /* Initialize js,je */

  pG_gpu->Nx2 = pG->Nx2;

  pG_gpu->js = pG->js;
  pG_gpu->je = pG->je;

  /* Compute dx2 */

  pG_gpu->dx2 = pG->dx2;

  /* Initialize j-displacement, and the x2-position of coordinate jx = 0. */

  pG_gpu->jdisp = pG->jdisp;
  pG_gpu->x2_0 = pG->x2_0;

  /* ---------  Allocate 2D arrays to hold Gas based on size of grid --------- */

  if (pG->Nx1 > 1)
    Nx1T = pG->Nx1 + 2*nghost;
  else
    Nx1T = 1;

  if (pG->Nx2 > 1)
    Nx2T = pG->Nx2 + 2*nghost;
  else
    Nx2T = 1;

  /* Allocate memory for gas structures and interface field (device memory) */
//  cout << "U: " << Nx2T*Nx1T << endl;


  /* U */
  code = cudaMalloc((void**)&pG_gpu->U, Nx2T*Nx1T*sizeof(Gas));
  if(code != cudaSuccess) {
    ath_error("[cudaMalloc_2d] U failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
        Nx1T, Nx2T, (int)(sizeof(Gas)), cudaGetErrorString(code));
  }
  /* B1i */
  code = cudaMalloc((void**)&pG_gpu->B1i, Nx2T*Nx1T*sizeof(Real));
  if(code != cudaSuccess) {
    cudaFree(pG_gpu->U);
    ath_error("[cudaMalloc_2d] B1i failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
            Nx1T, Nx2T, (int)(sizeof(Real)), cudaGetErrorString(code));
  }
  /* B2i */
  code = cudaMalloc((void**)&pG_gpu->B2i, Nx2T*Nx1T*sizeof(Real));
  if(code != cudaSuccess) {
    cudaFree(pG_gpu->U);
    cudaFree(pG_gpu->B1i);
    ath_error("[cudaMalloc_2d] B2i failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
        Nx1T, Nx2T, (int)(sizeof(Real)), cudaGetErrorString(code));
  }
  /* B3i */
  code = cudaMalloc((void**)&pG_gpu->B3i, Nx2T*Nx1T*sizeof(Real));
  if(code != cudaSuccess) {
    cudaFree(pG_gpu->U);
    cudaFree(pG_gpu->B1i);
    cudaFree(pG_gpu->B2i);
    ath_error("[cudaMalloc_2d] B3i failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
        Nx1T, Nx2T, (int)(sizeof(Real)), cudaGetErrorString(code));
  }

}

/**==============================================================================
 * Copy grid from CPU host memory to prepared GPU grid (also host memory).
 * It will copy U, and B1i, B2i, B3i from host memeory to device
 */
void copy_to_gpu_mem(Grid_gpu *pG_gpu, Grid *pG) {

  cudaError_t code;
  int i, Nx2T, Nx1T;

  /* Calculate physical size of grid */
  if (pG->Nx2 > 1)
    Nx2T = pG->Nx2 + 2*nghost;
  else
    Nx2T = 1;

  if (pG->Nx1 > 1)
    Nx1T = pG->Nx1 + 2*nghost;
  else
    Nx1T = 1;

  /* Start copying rows of gas variables from host to device memory */
  for(i=0; i<Nx2T; i++) {
    code = cudaMemcpy(pG_gpu->U+i*Nx1T+nghost, pG->U[i], sizeof(Gas)*(Nx1T-nghost), cudaMemcpyHostToDevice);
    if(code != cudaSuccess) {
      ath_error("[copy_to_gpu_mem U] error: %s\n", cudaGetErrorString(code));
    }
    code = cudaMemcpy(pG_gpu->B1i+i*Nx1T+nghost, pG->B1i[i], sizeof(Real)*(Nx1T-nghost), cudaMemcpyHostToDevice);
    if(code != cudaSuccess) {
      ath_error("[copy_to_gpu_mem B1i] error: %s\n", cudaGetErrorString(code));
    }
    code = cudaMemcpy(pG_gpu->B2i+i*Nx1T+nghost, pG->B2i[i], sizeof(Real)*(Nx1T-nghost), cudaMemcpyHostToDevice);
    if(code != cudaSuccess) {
      ath_error("[copy_to_gpu_mem B2i] error: %s\n", cudaGetErrorString(code));
    }
    code = cudaMemcpy(pG_gpu->B3i+i*Nx1T+nghost, pG->B3i[i], sizeof(Real)*(Nx1T-nghost), cudaMemcpyHostToDevice);
    if(code != cudaSuccess) {
      ath_error("[copy_to_gpu_mem B3i] error: %s\n", cudaGetErrorString(code));
    }
  }
}

/**==============================================================================
 * Copy whole structure form host memory to device memory
 */
void copy_gpu_to_gpu_mem(Grid_gpu *pG_gpu_dev, Grid_gpu *pG_host) {
  cudaError_t code;
  code = cudaMemcpy(pG_gpu_dev, pG_host, sizeof(Grid_gpu), cudaMemcpyHostToDevice);
  if(code != cudaSuccess) {
    ath_error("[copy_gpu_to_gpu_mem] error: %s\n", cudaGetErrorString(code));
  }
}

/**==============================================================================
 * Copy whole structure from device memory to host memory
 */
void copy_gpu_mem_to_gpu(Grid_gpu *pG_host, Grid_gpu *pG_gpu_dev) {
  cudaError_t code;
  code = cudaMemcpy(pG_host, pG_gpu_dev, sizeof(Grid_gpu), cudaMemcpyDeviceToHost);
  if(code != cudaSuccess) {
    ath_error("[copy_gpu_to_gpu_mem] error: %s\n", cudaGetErrorString(code));
  }
}

/**==============================================================================
 * Copy back grid structures from GPU device memory to
 * grid structure in host memory
 */
void copy_to_host_mem(Grid *pG, Grid_gpu *pG_gpu) {
  cudaError_t code;
  int i, Nx2T, Nx1T;

  if (pG->Nx2 > 1)
    Nx2T = pG->Nx2 + 2*nghost;
  else
    Nx2T = 1;

  if (pG->Nx1 > 1)
    Nx1T = pG->Nx1 + 2*nghost;
  else
    Nx1T = 1;

  /* Copy row by row */
  for(i=0; i<Nx2T; i++) {
    /* U */
    code = cudaMemcpy(pG->U[i], pG_gpu->U+i*Nx1T, sizeof(Gas)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
      ath_error("[copy_to_host_mem U] error: %s\n", cudaGetErrorString(code));
    }
    /* B1i */
    code = cudaMemcpy(pG->B1i[i], pG_gpu->B1i+i*Nx1T, sizeof(Real)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
      ath_error("[copy_to_host_mem B1i] error: %s\n", cudaGetErrorString(code));
    }
    /* B2i */
    code = cudaMemcpy(pG->B2i[i], pG_gpu->B2i+i*Nx1T, sizeof(Real)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
      ath_error("[copy_to_host_mem B2i] error: %s\n", cudaGetErrorString(code));
    }
    /* B3i */
    code = cudaMemcpy(pG->B3i[i], pG_gpu->B3i+i*Nx1T, sizeof(Real)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
      ath_error("[copy_to_host_mem B3i] error: %s\n", cudaGetErrorString(code));
    }
  }
}

