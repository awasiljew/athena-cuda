/*
 * integrate_2d_cuda.h
 * Mini how to:
 *      a) Define *.h file with extern definitions
 *      b) Define *.cu file with extern "C" definitions
 *      c) To use functions from *.h file, include this file in source
 *  Created on: 2010-02-25
 *      Author: adam
 */

#ifndef INTEGRATE_2D_CUDA_H_
#define INTEGRATE_2D_CUDA_H_

#include "../../athena.h"

extern void integrate_init_2d_cu(int nx1, int nx2);
extern void integrate_destruct_2d_cu();
extern void integrate_2d_cu(Grid_gpu *);

#endif /* INTEGRATE_2D_CUDA_H_ */
