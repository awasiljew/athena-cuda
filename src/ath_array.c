#include "copyright.h"
/*==============================================================================
 * FILE: ath_array.c
 *
 * PURPOSE: Functions that construct and destruct 1D, 2D and 3D arrays of any
 *   type. Elements in these arrays can be accessed in the standard fashion of
 *   array[in][im] (where in = 0 -> nr-1 and im = 0 -> nc-1) as if it
 *   were an array of fixed size at compile time with the statement:
 *      "Type" array[nr][nc];
 *  Equally so for 3D arrys:  array[nt][nr][nc]       -- TAG -- 8/2/2001
 *
 * EXAMPLE usage of 3D construct/destruct functions for arrays of type Real:
 *   array = (Real ***)calloc_3d_array(nt,nr,nc,sizeof(Real));
 *   free_3d_array(array);
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   calloc_1d_array() - creates 1D array
 *   calloc_2d_array() - creates 2D array
 *   calloc_3d_array() - creates 3D array
 *   free_1d_array()   - destroys 1D array
 *   free_2d_array()   - destroys 2D array
 *   free_3d_array()   - destroys 3D array
 *============================================================================*/

#include <stdlib.h>
#include "prototypes.h"
#include <cuda.h>
#include <cuda_runtime.h>

/*----------------------------------------------------------------------------*/
/* calloc_1d_array: construct 1D array = array[nc]  */

void* calloc_1d_array(size_t nc, size_t size)
{
  void *array;
  
  if ((array = (void *)calloc(nc,size)) == NULL) {
    ath_error("[calloc_1d] failed to allocate memory (%d of size %d)\n",
              (int)nc,(int)size);
    return NULL;
  }
  return array;
}

/*----------------------------------------------------------------------------*/
/* calloc_2d_array: construct 2D array = array[nr][nc]  */

void** calloc_2d_array(size_t nr, size_t nc, size_t size)
{
  void **array;
  size_t i;

  if((array = (void **)calloc(nr,sizeof(void*))) == NULL){
    ath_error("[calloc_2d] failed to allocate memory for %d pointers\n",(int)nr);
    return NULL;
  }

  if((array[0] = (void *)calloc(nr*nc,size)) == NULL){
    ath_error("[calloc_2d] failed to allocate memory (%d X %d of size %d)\n",
              (int)nr,(int)nc,(int)size);
    free((void *)array);
    return NULL;
  }

  for(i=1; i<nr; i++){
    array[i] = (void *)((unsigned char *)array[0] + i*nc*size);
  }

  return array;
}

/*----------------------------------------------------------------------------*/
/* calloc_3d_array: construct 3D array = array[nt][nr][nc]  */

void*** calloc_3d_array(size_t nt, size_t nr, size_t nc, size_t size)
{
  void ***array;
  size_t i,j;

  if((array = (void ***)calloc(nt,sizeof(void**))) == NULL){
    ath_error("[calloc_3d] failed to allocate memory for %d 1st-pointers\n",
              (int)nt);
    return NULL;
  }

  if((array[0] = (void **)calloc(nt*nr,sizeof(void*))) == NULL){
    ath_error("[calloc_3d] failed to allocate memory for %d 2nd-pointers\n",
              (int)(nt*nr));
    free((void *)array);
    return NULL;
  }

  for(i=1; i<nt; i++){
    array[i] = (void **)((unsigned char *)array[0] + i*nr*sizeof(void*));
  }

  if((array[0][0] = (void *)calloc(nt*nr*nc,size)) == NULL){
    ath_error("[calloc_3d] failed to alloc. memory (%d X %d X %d of size %d)\n",
              (int)nt,(int)nr,(int)nc,(int)size);
    free((void *)array[0]);
    free((void *)array);
    return NULL;
  }

  for(j=1; j<nr; j++){
    array[0][j] = (void **)((unsigned char *)array[0][j-1] + nc*size);
  }

  for(i=1; i<nt; i++){
    array[i][0] = (void **)((unsigned char *)array[i-1][0] + nr*nc*size);
    for(j=1; j<nr; j++){
      array[i][j] = (void **)((unsigned char *)array[i][j-1] + nc*size);
    }
  }

  return array;
}

/*----------------------------------------------------------------------------*/
/* free_1d_array: free memory used by 1D array  */

void free_1d_array(void *array)
{
  free(array);
}

/*----------------------------------------------------------------------------*/
/* free_2d_array: free memory used by 2D array  */

void free_2d_array(void *array)
{
  void **ta = (void **)array;

  free(ta[0]);
  free(array);
}

/*----------------------------------------------------------------------------*/
/* free_3d_array: free memory used by 3D array  */

void free_3d_array(void *array)
{
  void ***ta = (void ***)array;

  free(ta[0][0]);
  free(ta[0]);
  free(array);
}

/*----------------------------------------------------------------------------*/
/*-------------------------------   CUDA   -----------------------------------*/
/*----------------------------------------------------------------------------*/

/**==============================================================================
 * Allocate memory on device
 */
int calloc_1d_array_cu(void **array, size_t nc, size_t size) {
  cudaError_t code;
  if((code = cudaMalloc((void**)&array, nc*size)) != cudaSuccess) {
    ath_error("[calloc_1d] failed to allocate memory (%d of size %d) -> %s\n",
        (int)nc,(int)size, cudaGetErrorString(code));
    return 0;
  }
  return 1;
}

/**==============================================================================
 * Allocate memory on device. 2-dimensional array is allocated as 1D with
 * appriopriate size (size == row*cols).
 */
int calloc_2d_array_cu(void **array, size_t nr, size_t nc, size_t size) {
  cudaError_t code;
  if(code = (cudaMalloc((void**)&array, nr*nc*size)) != cudaSuccess) {
    ath_error("[calloc_2d] failed to allocate memory for (%d x %d of size %d) pointers -> %s\n",
        (int)nr, (int)nc, (int)size, cudaGetErrorString(code));
    return 0;
  }
  return 1;
}

/**==============================================================================
 * deprecated
 */
void free_1d_array_cu(void *array) {
//  cudaError_t code;
//  if((code = cudaFree(array)) != cudaSuccess) {
//    printf("free_1d_array_cu %s\n", cudaGetErrorString(code));
//  }
}

/**==============================================================================
 * deprecated
 */
void free_2d_array_cu(void *array) {
//  cudaError_t code;
//  if((code = cudaFree(array)) != cudaSuccess) {
//    printf("free_2d_array_cu %s\n", cudaGetErrorString(code));
//  }
}
