/* 
 * File:   main.c
 * Author: adam
 *
 * Created on 27 listopad 2009, 11:19
 */
#define MAIN_C

#include <stdio.h>
#include <stdlib.h>
#include "athena.h"
#include "defs.h"
#include "globals.h"
#include "prototypes.h"
#include "cuda_engine/include/integrate_2d_cuda.h"
#include "cuda_engine/include/set_bvals_cuda.h"
#include "debug_tools_cuda.h"
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

void usage(char *prog);

/*
 * Main program
 */
int main(int argc, char** argv) {

  /* Error code handler */
  cudaError_t code;

  /* Level 0 gird domain - used to compute on CPU */
  Grid level0_Grid;

  /* Level 0 grids domain for GPU - device */
  Grid_gpu level0_grid_gpu_dev;

#ifdef GPU_HOST_COMPARE

  /* Level 1 grid domain - used to compare results between GPU and CPU */
  Grid level1_Grid;

#endif /* GPU_HOST_COMPARE */

  char *definput = "athinput", *rundir = NULL;
  char *athinput = definput;

  int i;

  int iflush = 1;

  /*--- Step 1. ----------------------------------------------------------------*/
  /* Check for command line options and respond.  See comments in usage()
   * for description of options.  */

  for (i = 1; i < argc; i++) {
      if (*argv[i] == '-' && *(argv[i] + 1) != '\0' && *(argv[i] + 2) == '\0') {
          switch (*(argv[i] + 1)) {
          case 'i': /* -i <file>   */
            athinput = argv[++i];
            break;
          case 'd': /* -d <directory>   */
            rundir = argv[++i];
            break;
          case 'h': /* -h */
            usage(argv[0]);
            break;
          default:
            usage(argv[0]);
            break;
          }
      }
  }

  par_open(athinput); /* opens AND reads */
  par_cmdline(argc, argv);
  show_config_par(); /* Add the configure block to the parameter database */

  struct timeval t1, t2, t1_gpu, t2_gpu;

  /*--- Step 3. ----------------------------------------------------------------*/
  /* set variables in <time> block (these control execution time) */

  CourNo = par_getd("time", "cour_no");
  int nlim = par_geti_def("time", "nlim", -1);
  Real tlim = par_getd("time", "tlim");

  /* Set variables in <job> block, EOS parameters from <problem> block.  */

  level0_Grid.outfilename = par_gets("job", "problem_id");

  Gamma = par_getd("problem", "gamma");
  Gamma_1 = Gamma - 1.0;
  Gamma_2 = Gamma - 2.0;

  /* Initialize grid on host memory, it should be copied to device memory.
   * Initialization means that memory will be allocated for all
   * physical variables and additional properties (such as dimensions) will
   * be set */
  init_grid(&level0_Grid);

#ifdef GPU_HOST_COMPARE

  init_grid(&level1_Grid);

#endif /* GPU_HOST_COMPARE */

#ifndef ONLY_HOST

  /**
   * Initializes grid based on host grid. Makes a copy of structure from host
   * level 0 grid to level 0 grid for GPU (also host memory), to a level 0
   * grid for GPU on device memory. The latter (first parameter of below function)
   * is the grid on which kernel GPU functions will perform computations
   */
  init_grid_gpu(&level0_grid_gpu_dev, &level0_Grid);

#endif /* ONLY_HOST */

  /**
   * Initialize problem over host grid (based on current implementation). Initial problem will
   * be copied to device memory later.
   */
  problem(&level0_Grid);

  /*--- Step 4. ----------------------------------------------------------------*/
  /* set boundary value function pointers using BC flags in <grid> blocks, then
   * set boundary conditions for initial conditions  */

  set_bvals_init(&level0_Grid);

#ifndef ONLY_HOST

  set_bvals_init_cu(&level0_grid_gpu_dev, par_geti("grid","ibc_x1"), par_geti("grid","obc_x1"), par_geti("grid","ibc_x2"), par_geti("grid","obc_x2"));

#endif /* ONLY_HOST */

  /* Only bvals for Gas structure set when last argument of set_bvals = 0  */
  set_bvals(&level0_Grid, 0);

#ifndef ONLY_HOST

  /* Setup boundary condition on device grid */
  set_bvals_cu(&level0_grid_gpu_dev, 0);

#endif /* ONLY_HOST */

  gettimeofday(&t1, 0); // start
  new_dt(&level0_Grid);
  gettimeofday(&t2, 0); // stop

#ifndef ONLY_HOST

  /* GPU CODE - new dt */
  dt_init(&level0_grid_gpu_dev);

  /**
   * Make a copy of grid with initial conditions -> to GPU host memory
   */
  copy_to_gpu_mem(&level0_grid_gpu_dev, &level0_Grid);

  /* Setup boundary condition on device grid */
  set_bvals_cu(&level0_grid_gpu_dev, 0);

  /* Calculate new time step */

  gettimeofday(&t1_gpu, 0); // start
  new_dt_cuda(&level0_grid_gpu_dev, Gamma, Gamma_1, CourNo);
  gettimeofday(&t2_gpu, 0); // stop

#ifdef GPU_HOST_COMPARE

  printTimeSpeedup(t1, t2, t1_gpu, t2_gpu,"new_dt");

#endif /* GPU_HOST_COMPARE */
#endif /* ONLY_HOST */


  /*--- Step 5. ----------------------------------------------------------------*/
  /* Set output modes (based on <ouput> blocks in input file).
   * Allocate temporary arrays needed by solver */

  init_output(&level0_Grid);

#ifdef GPU_HOST_COMPARE

  lr_states_init(level0_Grid.Nx1, level0_Grid.Nx2);

  integrate_init_2d(level0_Grid.Nx1, level0_Grid.Nx2);

#endif /* GPU_HOST_COMPARE */

#ifdef ONLY_HOST

  lr_states_init(level0_Grid.Nx1, level0_Grid.Nx2);

  integrate_init_2d(level0_Grid.Nx1, level0_Grid.Nx2);

#else

  /* LR states init on GPU */
  lr_states_init_cu(level0_Grid.Nx1, level0_Grid.Nx2);

  /* Allocate work arrays on GPU */
  integrate_init_2d_cu(level0_Grid.Nx1, level0_Grid.Nx2);

#endif /* ONLY_HOST */

  ath_pout(0, "\nSetup complete, entering main loop...\n\n");
  ath_pout(0, "cycle=%i time=%e dt=%e\n", level0_Grid.nstep, level0_Grid.time,
      level0_Grid.dt);

  /*--- Step 6. ---------------------------------------------------------------*/
  /* START OF MAIN INTEGRATION LOOP ==============================================
   * Steps are: (1) Check for data ouput
   *            (2) Integrate level0 grid
   *            (3) Set boundary values
   *            (4) Set new timestep
   */

  while (level0_Grid.time < tlim && (nlim < 0 || level0_Grid.nstep < nlim)) {

      /* Only write output's with t_out>t when last argument of data_output = 0 */
      /* Before data output copy data from GPU memory to HOST memory, only when its needed (that advance exceeded dt for output.
       * Do not copy whole structure each time step, because it will consume a lot of time */
      data_output(&level0_Grid, &level0_grid_gpu_dev, 0);
 //     break;

      /* modify timestep so loop finishes at t=tlim exactly */
      if ((tlim - level0_Grid.time) < level0_Grid.dt) {
          level0_Grid.dt = (tlim - level0_Grid.time);
      }

#ifdef GPU_HOST_COMPARE

      /* Copy result from GPU to host memory to get error information */
      //copy_to_gpu_mem(&level0_grid_gpu_dev, &level1_Grid);

      /* Main integration algorithm */
      gettimeofday(&t1, 0); // start
      integrate_2d(&level0_Grid);
      gettimeofday(&t2, 0); // stop

#endif /* GPU_HOST_COMPARE */

#ifdef ONLY_HOST

      /* Main integration algorithm */
      integrate_2d(&level0_Grid);

#else

      gettimeofday(&t1_gpu, 0); // start
      integrate_2d_cuda(&level0_grid_gpu_dev);
      gettimeofday(&t2_gpu, 0); // stop

#ifdef GPU_HOST_COMPARE

      printTimeSpeedup(t1, t2, t1_gpu, t2_gpu,"integrate_2d");

#endif /* GPU_HOST_COMPARE */

#endif /* ONLY_HOST */


#ifdef ONLY_HOST

      level0_Grid.nstep++;
      level0_Grid.time += level0_Grid.dt;

      new_dt(&level0_Grid);

#else

//      copy_to_host_mem(&level0_Grid, &level0_grid_gpu_dev);
////
////
////
//            level0_Grid.nstep++;
//            level0_Grid.time += level0_Grid.dt;
////
//            new_dt(&level0_Grid);
//
//            level0_grid_gpu_dev.dt = level0_Grid.dt;
//            level0_grid_gpu_dev.time = level0_Grid.time;
//            level0_grid_gpu_dev.nstep = level0_Grid.nstep;


      level0_grid_gpu_dev.nstep++;
      level0_grid_gpu_dev.time += level0_grid_gpu_dev.dt;

      new_dt_cuda(&level0_grid_gpu_dev, Gamma, Gamma_1, CourNo);

      level0_Grid.dt = level0_grid_gpu_dev.dt;
      level0_Grid.time = level0_grid_gpu_dev.time;
      level0_Grid.nstep = level0_grid_gpu_dev.nstep;

      /* Boundary values must be set after time is updated for t-dependent BCs
       * Only bvals for Gas structure set when last argument of set_bvals = 0  */

      set_bvals_cu(&level0_grid_gpu_dev, 0);

#endif /* ONLY_HOST */

#ifdef ONLY_HOST

      set_bvals(&level0_Grid, 0);

#endif /* ONLY_HOST */

      ath_pout(0, "cycle=%i time=%e dt=%e\n", level0_Grid.nstep,
          level0_Grid.time, level0_Grid.dt);

      if (iflush) {
          ath_flush_out();
          ath_flush_err();
      }
  }
  /* END OF MAIN INTEGRATION LOOP ==============================================*/

  ath_pout(0, "\nEnd of main loop\n\n");

  lr_states_destruct();
  integrate_destruct_2d();

#ifndef ONLY_HOST

  integrate_destruct_2d_cu();

#endif

  data_output_destruct();
  par_close();
  dt_destruct();

#ifndef ONLY_HOST
  code = cudaFree(level0_grid_gpu_dev.U);
  printf("cudaFree: %s\n", cudaGetErrorString(code));
  code = cudaFree(level0_grid_gpu_dev.B1i);
  printf("cudaFree: %s\n", cudaGetErrorString(code));
  code = cudaFree(level0_grid_gpu_dev.B2i);
  printf("cudaFree: %s\n", cudaGetErrorString(code));
  code = cudaFree(level0_grid_gpu_dev.B3i);
  printf("cudaFree: %s\n", cudaGetErrorString(code));
#endif /* ONLY_HOST */

  return (EXIT_SUCCESS);
}

/*----------------------------------------------------------------------------*/
/*  usage: outputs help
 *    athena_version is hardwired at beginning of this file
 *    CONFIGURE_DATE is macro set when configure script runs */

void
usage(char *prog)
{
  ath_perr(-1, "\nUsage: %s [options] [block/par=value ...]\n", prog);
  ath_perr(-1, "\nOptions:\n");
  ath_perr(-1, "  -i <file>       Alternate input file [athinput]\n");
  ath_perr(-1, "  -d <directory>  Alternate run dir [current dir]\n");
  ath_perr(-1, "  -h              This Help, and configuration settings\n");
  ath_perr(-1, "  -n              Parse input, but don't run program\n");
  exit(0);
}
