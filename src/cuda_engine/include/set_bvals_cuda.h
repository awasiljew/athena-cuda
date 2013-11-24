/*
 * set_bvals_cuda.h
 *
 *  Created on: 2010-03-04
 *      Author: adam
 */

#ifndef SET_BVALS_CUDA_H_
#define SET_BVALS_CUDA_H_

#include "../../athena.h"

extern void set_bvals_cu(Grid_gpu *pGrid, int var_flag);

#endif /* SET_BVALS_CUDA_H_ */
