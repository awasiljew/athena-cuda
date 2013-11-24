#ifndef PROTOTYPES_H
#define PROTOTYPES_H 
#include "copyright.h"
/*==============================================================================
 * FILE: prototypes.h
 *
 * PURPOSE: Prototypes for all public functions from the following files:
 *   main.c
 *   ath_array.c
 *   ath_files.c
 *   ath_log.c
 *   ath_signal.c
 *   cc_pos.c
 *   convert_var.c
 *   esystem_prim.c, esystem_roe.c
 *   flux_force.c, flux_hllc.c, flux_hlld.c, flux_hlle.c, flux_roe.c, 
 *     flux_2shock.c, flux_exact
 *   init_domain.c
 *   init_grid.c
 *   integrate.c
 *   integrate_1d.c, integrate_2d.c, integrate_3d-vl., integrate_3d-ctu.c
 *   lr_states_prim1.c, lr_states_prim2.c, lr_states_prim3.c
 *   new_dt.c
 *   output.c
 *   output_fits.c, output_pdf.c output_pgm.c, output_ppm.c, output_tab.c
 *   dump_binary.c, dump_dx.c, dump_history.c, dump_table.c, dump_vtk.c
 *   par.c
 *   restart.c
 *   set_bvals.c
 *   show_config.c
 *   utils.c
 *============================================================================*/

#include <stdio.h>
#include <stdarg.h>
#include "athena.h"
#include "defs.h"

/*----------------------------------------------------------------------------*/
/* main.c */
int athena_main(int argc, char *argv[]);

/*----------------------------------------------------------------------------*/
/* ath_array.c */
void*   calloc_1d_array(                      size_t nc, size_t size);
void**  calloc_2d_array(           size_t nr, size_t nc, size_t size);
void*** calloc_3d_array(size_t nt, size_t nr, size_t nc, size_t size);
void free_1d_array(void *array);
void free_2d_array(void *array);
void free_3d_array(void *array);

/* ath_array.c - cuda functions */
int calloc_1d_array_cu(void **array,            size_t nc, size_t size);
int calloc_2d_array_cu(void **array, size_t nr, size_t nc, size_t size);
//void*** calloc_3d_array_cu(size_t nt, size_t nr, size_t nc, size_t size);
void free_1d_array_cu(void *array);
void free_2d_array_cu(void *array);
//void free_3d_array_cu(void *array);

/*----------------------------------------------------------------------------*/
/* ath_files.c */
char *fname_construct(const char *basename, const int dlen, const int idump, 
		      const char *type, const char *ext);
FILE *ath_fopen(const char *basename, const int dlen, const int idump, 
		const char *type, const char *ext, const char *mode);
size_t ath_fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);

/*----------------------------------------------------------------------------*/
/* cc_pos.c */
void cc_pos(const Grid *pG, const int i, const int j,
            Real *px1, Real *px2);

/*----------------------------------------------------------------------------*/
/* convert_var.c */
void Cons1D_to_Prim1D(const Cons1D *pU, Prim1D *pW, const Real *pBx);
void Prim1D_to_Cons1D(Cons1D *pU, const Prim1D *pW, const Real *pBx);
Real cfast(const Cons1D *U, const Real *Bx);

/*----------------------------------------------------------------------------*/
/* esystem_*.c */
void esys_prim_adb_mhd(const Real d, const Real v1, const Real p,
  const Real b1, const Real b2, const Real b3, Real eigenvalues[],
  Real right_eigenmatrix[][7], Real left_eigenmatrix[][7]);

void esys_roe_adb_mhd(const Real d, const Real v1, const Real v2,
  const Real v3, const Real h, const Real b1, const Real b2, const Real b3,
  const Real x, const Real y, Real eigenvalues[],
  Real right_eigenmatrix[][7], Real left_eigenmatrix[][7]);

/*----------------------------------------------------------------------------*/
/* flux_*.c */
void flux_hlle  (const Cons1D Ul,const Cons1D Ur,
                 const Prim1D Wl,const Prim1D Wr,const Real Bxi, Cons1D *pF);
void flux_roe   (const Cons1D Ul,const Cons1D Ur,
                 const Prim1D Wl,const Prim1D Wr,const Real Bxi, Cons1D *pF);

/*----------------------------------------------------------------------------*/
/* init_grid.c */
void init_grid(Grid *pGrid);

/*----------------------------------------------------------------------------*/
/* integrate_2d.c */
void integrate_destruct_2d(void);
void integrate_init_2d(int Nx1, int Nx2);
void integrate_2d(Grid *pgrid);
/* integrate_2d.c - CUDA */
//void integrate_init_2d_cu(int nx1, int nx2);
//void integrate_destruct_2d_cu(void);

/*----------------------------------------------------------------------------*/
/* lr_states.c */
void lr_states_destruct(void);
void lr_states_init(int nx1, int nx2);
void lr_states(const Prim1D W[], const Real Bxc[],
               const Real dt, const Real dtodx, const int is, const int ie,
               Prim1D Wl[], Prim1D Wr[]);

/*----------------------------------------------------------------------------*/
/* new_dt.c */
void new_dt(Grid *pGrid);

/*----------------------------------------------------------------------------*/
/* output.c - and related files */
void init_output(Grid *pGrid);
void data_output(Grid *pGrid, Grid_gpu *pGpuGrid, const int flag);
int  add_output(Output *new_out);
void add_rst_out(Output *new_out);
void data_output_destruct(void);
void data_output_enroll(Real time, Real dt, int num, const VGFunout_t fun,
	                const char *fmt,  const Gasfun_t expr, int n,
                        const Real dmin, const Real dmax, int sdmin, int sdmax);
void dump_history_enroll(const Gasfun_t pfun, const char *label);
float  **subset2(Grid *pGrid, Output *pout);

void output_ppm  (Grid *pGrid, Output *pOut);
void output_tab(Grid *pGrid, Output *pOut);
void dump_binary (Grid *pGrid, Output *pOut);
void dump_tab (Grid *pGrid, Output *pOut);

/*----------------------------------------------------------------------------*/
/* par.c */
void   par_open(char *filename);
void   par_cmdline(int argc, char *argv[]);
int    par_exist(char *block, char *name);

char  *par_gets(char *block, char *name);
int    par_geti(char *block, char *name);
double par_getd(char *block, char *name);

char  *par_gets_def(char *block, char *name, char   *def);
int    par_geti_def(char *block, char *name, int    def);
double par_getd_def(char *block, char *name, double def);

void   par_sets(char *block, char *name, char *sval, char *comment);
void   par_seti(char *block, char *name, char *fmt, int ival, char *comment);
void   par_setd(char *block, char *name, char *fmt, double dval, char *comment);

void   par_dump(int mode, FILE *fp);
void   par_close(void);


/*----------------------------------------------------------------------------*/
/* prob/PROBLEM.c ; linked to problem.c */
void problem(Grid *pgrid);
void Userwork_in_loop(Grid *pgrid);
void Userwork_after_loop(Grid *pgrid);
Gasfun_t get_usr_expr(const char *expr);


/*----------------------------------------------------------------------------*/
/* set_bvals.c  */
void set_bvals_init(Grid *pG);
void set_bvals_start(VGFun_t start);
void set_bvals_fun(enum Direction dir, VBCFun_t prob_bc);
void set_bvals(Grid *pGrid, int var_flag);

/*----------------------------------------------------------------------------*/
/* show_config.c */
void show_config(void);
void show_config_par(void);

/*----------------------------------------------------------------------------*/
/* utils.c */
char *ath_strdup(const char *in);
int ath_gcd(int a, int b);
int ath_big_endian(void);
void ath_bswap(void *vdat, int sizeof_len, int cnt);
void ath_error(char *fmt, ...);
void minmax1(float *data, int nx1, float *dmin, float *dmax);
void minmax2(float **data, int nx2, int nx1, float *dmin, float *dmax);
void minmax3(float ***data, int nx3, int nx2, int nx1,float *dmin,float *dmax);
FILE *atherr_fp(void);

/*----------------------------------------------------------------------------*/
/* utils_cu.c */
//extern
//void init_grid_gpu(Grid_gpu *pG_gpu, Grid *pG);
//extern
//void copy_to_gpu_mem(Grid_gpu *pG_gpu, Grid *pG);
//extern
//void copy_gpu_to_gpu_mem(Grid_gpu *pG_gpu_dev, Grid_gpu *pG_host);
//extern
//void copy_to_host_mem(Grid *pG, Grid_gpu *pG_gpu);
//extern
//void copy_gpu_mem_to_gpu(Grid_gpu *pG_host, Grid_gpu *pG_gpu_dev);
//void copu_to_gpu_mem(Grid *pG, Grid *pG_gpu);

#endif /* PROTOTYPES_H */
