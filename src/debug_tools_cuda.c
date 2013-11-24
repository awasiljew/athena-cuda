#include "debug_tools_cuda.h"
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*************************************************************************/
void setup_is_ie(int *is, int *ie, int n, int lo, int hi) {
	if(lo == hi == 0) {
		*is = lo;
		*ie = hi;
	} else {
		*is = 0;
		*ie = n-1;
	}
}

/*************************************************************************/
int compare_reals(Real r1, Real r2) {
	if ((isnan(r1) && isnan(r2)) || fabs(r1 - r2) <= ACCURACY) {
		return 1;
	}
#ifdef REALS_CMP_DETAILS
	printf("Reals compare %.12e - %.12e = % e\n", r1, r2, fabs(r1 - r2));
#endif
	return 0;
}

/*************************************************************************/
int compare_reals_array(Real *r1, Real* r2, int n, int lo, int hi) {
	int i, is, ie, j = 1, k;
	setup_is_ie(&is, &ie, n, lo, hi);

	for(i=is; i<=ie; i++) {
		k = compare_reals(r1[i], r2[i]);

#ifdef REALS_CMP_ARR
		if(!k) printf("Diff Reals Array [%d] %e %e\n", i, r1[i], r2[i]);
#endif

		j = j && k;

	}

	return j;
}

/*************************************************************************/
int compare_gas(Gas g1, Gas g2) {
	int B1c = compare_reals(g1.B1c, g2.B1c);
	int B2c = compare_reals(g1.B2c, g2.B2c);
	int B3c = compare_reals(g1.B3c, g2.B3c);
	int E = compare_reals(g1.E, g2.E);
	int M1 = compare_reals(g1.M1, g2.M1);
	int M2 = compare_reals(g1.M2, g2.M2);
	int M3 = compare_reals(g1.M3, g2.M3);
	int d = compare_reals(g1.d, g2.d);

#ifdef GAS_CMP_DETAILS

	if(!B1c) printf("B1c diff %e %e\n",g1.B1c, g2.B1c);
	if(!B2c) printf("B2c diff %e %e\n",g1.B2c, g2.B2c);
	if(!B3c) printf("B3c diff %e %e\n",g1.B3c, g2.B3c);
	if(!E) printf("E diff %e %e\n",g1.E, g2.E);
	if(!M1) printf("M1 diff %e %e\n",g1.M1, g2.M1);
	if(!M2) printf("M2 diff %e %e\n",g1.M2, g2.M2);
	if(!M3) printf("M3 diff %e %e\n",g1.M3, g2.M3);
	if(!d) printf("d diff %e %e\n",g1.d, g2.d);

#endif

	return B1c && B2c && B3c && E && M1 && M2 && M3 && d;
}

/*************************************************************************/
int compare_gas_array(Gas* g1, Gas *g2, int n, int lo, int hi) {
	int i, is, ie, j = 1, k;
	setup_is_ie(&is, &ie, n, lo, hi);

	for(i = is; i<=ie; i++) {
		k = compare_gas(g1[i], g2[i]);

#ifdef GAS_CMP_DETAILS_ARR
		if(!k) printf("Diff Gas Array [%d]\n", i);
#endif

		j = j && k;
	}

	return j;
}

/*************************************************************************/
int compare_Prim1D(Prim1D p1, Prim1D p2) {
	int By = compare_reals(p1.By, p2.By);
	int Bz = compare_reals(p1.Bz, p2.Bz);
	int P = compare_reals(p1.P, p2.P);
	int Vx = compare_reals(p1.Vx, p2.Vx);
	int Vz = compare_reals(p1.Vz, p2.Vz);
	int d = compare_reals(p1.d, p2.d);

#ifdef PRIM1D_CMP_DETAILS

	if(!By) printf("By diff %e %e\n",p1.By, p2.By);
	if(!Bz) printf("Bz diff %e %e\n",p1.Bz, p2.Bz);
	if(!P) printf("P diff %e %e\n",p1.P, p2.P);
	if(!Vx) printf("Vx diff %e %e\n",p1.Vx, p2.Vx);
	if(!Vz) printf("Vz diff %e %e\n",p1.Vz, p2.Vz);
	if(!d) printf("d diff %e %e\n",p1.d, p2.d);

#endif

	return By && Bz && P && Vx && Vz && d;
}

/*************************************************************************/
int compare_Prim1D_array(Prim1D* p1, Prim1D* p2, int n, int lo, int hi) {
	int i, is, ie, j = 1, k;
	setup_is_ie(&is, &ie, n, lo, hi);

	for(i = is; i<=ie; i++) {
		k = compare_Prim1D(p1[i], p2[i]);

#ifdef PRIM1D_CMP_DETAILS_ARR
		if(!k) printf("Diff Prim1D Array [%d]\n", i);
#endif

		j = j && k;

	}

	return j;
}

/*************************************************************************/
int compare_Cons1D(Cons1D c1, Cons1D c2) {
	int By = compare_reals(c1.By, c2.By);
	int Bz = compare_reals(c1.Bz, c2.Bz);
	int E = compare_reals(c1.E, c2.E);
	int Mx = compare_reals(c1.Mx, c2.Mx);
	int My = compare_reals(c1.My, c2.My);
	int Mz = compare_reals(c1.Mz, c2.Mz);
	int d = compare_reals(c1.d, c2.d);

#ifdef CONS1D_CMP_DETAILS

	if(!By) printf("By diff %e %e\n",c1.By, c2.By);
	if(!Bz) printf("Bz diff %e %e\n",c1.Bz, c2.Bz);
	if(!E) printf("E diff %e %e\n",c1.E, c2.E);
	if(!Mx) printf("Mx diff %e %e\n",c1.Mx, c2.Mx);
	if(!My) printf("My diff %e %e\n",c1.My, c2.My);
	if(!Mz) printf("Mz diff %e %e\n",c1.Mz, c2.Mz);
	if(!d) printf("d diff %e %e\n",c1.d, c2.d);

#endif

	return By && Bz && E && Mx && My && Mz && d;
}

/*************************************************************************/
int compare_Cons1D_array(Cons1D *c1, Cons1D *c2, int n, int lo, int hi) {
	int i, is, ie, j = 1, k;
	setup_is_ie(&is, &ie, n, lo, hi);

	for(i = is; i<=ie; i++) {
		k = compare_Cons1D(c1[i], c2[i]);

#ifdef CONS1D_CMP_DETAILS_ARR
		if(!k) printf("Diff Cons1D Array [%d]\n", i);
#endif

		j = j && k;

	}

	return j;
}

/*************************************************************************/
int compare_grid_cpu(Grid* g1, Grid* g2, int ghost) {
	int is, ie;
	int js, je;
	int i,j;
	int result = 1;

	if(!ghost) {
		is = g1->is;
		ie = g1->ie;
		js = g1->js;
		je = g1->je;
	} else {
		is = 0;
		ie = g1->Nx1+2*nghost-1;
		js = 0;
		je = g1->Nx2+2*nghost-1;
	}

	for(j = js; j<=je; j++) {
		i = compare_gas_array(g1->U[j], g2->U[j], 0, is, ie);
		result = result && i;

#ifdef GRID_CMP_DETAILS_U
		if(!i) printf("Diff Grid Array U [%d][*]\n", j);
#endif

		i = compare_reals_array(g1->B1i[j], g2->B1i[j], 0, is, ie);
		result = result && i;

#ifdef GRID_CMP_DETAILS_B1I
		if(!i) printf("Diff Grid Array B1i [%d][*]\n", j);
#endif

		i = compare_reals_array(g1->B2i[j], g2->B2i[j], 0, is, ie);
		result = result && i;

#ifdef GRID_CMP_DETAILS_B2I
		if(!i) printf("Diff Grid Array B2i [%d][*]\n", j);
#endif

		i = compare_reals_array(g1->B3i[j], g2->B3i[j], 0, is, ie);
		result = result && i;

#ifdef GRID_CMP_DETAILS_B3I
		if(!i) printf("Diff Grid Array B3i [%d][*]\n", j);
#endif

	}

	return i;
}

/*************************************************************************/
/**==============================================================================
 * Copy back grid structures from GPU device memory to
 * grid structure in host memory
 */
void copy_to_host_mem_private(Grid *pG, Grid_gpu *pG_gpu) {
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
      fprintf(stderr,"[copy_to_host_mem U] error: %s\n", cudaGetErrorString(code));
    }
    /* B1i */
    code = cudaMemcpy(pG->B1i[i], pG_gpu->B1i+i*Nx1T, sizeof(Real)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
    	fprintf(stderr,"[copy_to_host_mem B1i] error: %s\n", cudaGetErrorString(code));
    }
    /* B2i */
    code = cudaMemcpy(pG->B2i[i], pG_gpu->B2i+i*Nx1T, sizeof(Real)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
    	fprintf(stderr,"[copy_to_host_mem B2i] error: %s\n", cudaGetErrorString(code));
    }
    /* B3i */
    code = cudaMemcpy(pG->B3i[i], pG_gpu->B3i+i*Nx1T, sizeof(Real)*Nx1T, cudaMemcpyDeviceToHost);
    if(code != cudaSuccess) {
    	fprintf(stderr,"[copy_to_host_mem B3i] error: %s\n", cudaGetErrorString(code));
    }
  }
}


/*************************************************************************/
int compare_grid_gpu(Grid_gpu* g1, Grid* g2, Grid* workGrid, int ghost) {
	copy_to_host_mem_private(workGrid, g1);
	return compare_grid_cpu(g2, workGrid, ghost);
}
