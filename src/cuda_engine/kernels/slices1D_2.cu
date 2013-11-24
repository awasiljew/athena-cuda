//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

__global__ void Cons1D_to_Prim1D_Slice1D_8b(Cons1D *Ul_x1Face_dev, Cons1D *Ur_x1Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B1_x1Face_dev, Cons1D *x1Flux_dev, int is, int ie, int js, int je, int sizex, Real Gamma_1, Real Gamma_2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j;
  calculateIndexes2D(&i, &j, sizex);

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  /* Main algorithm */
  Cons1D_to_Prim1D_cu_dev(&Ul_x1Face_dev[ind],&Wl_dev[ind],&B1_x1Face_dev[ind], Gamma_1);
  Cons1D_to_Prim1D_cu_dev(&Ur_x1Face_dev[ind],&Wr_dev[ind],&B1_x1Face_dev[ind], Gamma_1);

  flux_roe_cu_dev(Ul_x1Face_dev[ind],Ur_x1Face_dev[ind],Wl_dev[ind],Wr_dev[ind],
                 B1_x1Face_dev[ind],&x1Flux_dev[ind], Gamma_1, Gamma_2);

//  for (j=js-1; j<=je+1; j++) {
//      for (i=is; i<=ie+1; i++) {
//
//        Cons1D_to_Prim1D(&Ul_x1Face[j][i],&Wl[i],&B1_x1Face[j][i]);
//        Cons1D_to_Prim1D(&Ur_x1Face[j][i],&Wr[i],&B1_x1Face[j][i]);
//
//        flux_roe(Ul_x1Face[j][i],Ur_x1Face[j][i],Wl[i],Wr[i],
//                   B1_x1Face[j][i],&x1Flux[j][i]);
//      }
//    }
}

__global__ void Cons1D_to_Prim1D_Slice1D_8b_next(Cons1D *Ul_x1Face_dev, Cons1D *Ur_x1Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B1_x1Face_dev, Cons1D *x1Flux_dev, int is, int ie, int js, int je, int sizex, Real Gamma_1, Real Gamma_2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  flux_roe_cu_dev(Ul_x1Face_dev[ind],Ur_x1Face_dev[ind],Wl_dev[i],Wr_dev[i],
                 B1_x1Face_dev[ind],&x1Flux_dev[ind], Gamma_1, Gamma_2);

//  for (j=js-1; j<=je+1; j++) {
//      for (i=is; i<=ie+1; i++) {
//
//        Cons1D_to_Prim1D(&Ul_x1Face[j][i],&Wl[i],&B1_x1Face[j][i]);
//        Cons1D_to_Prim1D(&Ur_x1Face[j][i],&Wr[i],&B1_x1Face[j][i]);
//
//        flux_roe(Ul_x1Face[j][i],Ur_x1Face[j][i],Wl[i],Wr[i],
//                   B1_x1Face[j][i],&x1Flux[j][i]);
//      }
//    }
}

// Started as:
// Cons1D_to_Prim1D_Slice1D_8c<<<nnBlocks, BLOCK_SIZE>>>(Ul_x2Face_dev, Ur_x2Face_dev, Wl_dev, Wr_dev, B2_x2Face_dev, x2Flux_dev, is-1, ie+1, js, je+1, sizex, Gamma_1, Gamma_2);
__global__ void Cons1D_to_Prim1D_Slice1D_8c(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B2_x2Face_dev, Cons1D *x2Flux_dev, int is, int ie, int js, int je, int sizex, Real Gamma_1, Real Gamma_2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j;
  calculateIndexes2D(&i, &j, sizex);

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

//  x2Flux_dev[ind].By = 0.0;
//  x2Flux_dev[ind].Bz = 0.0;
//  x2Flux_dev[ind].E = 0.0;
//  x2Flux_dev[ind].Mx = 0.0;
//  x2Flux_dev[ind].My = 0.0;
//  x2Flux_dev[ind].Mz = 0.0;
//  x2Flux_dev[ind].d = 0.0;
//
//  return;

  /* Main algorithm */
  Cons1D_to_Prim1D_cu_dev(&Ul_x2Face_dev[ind],&Wl_dev[ind],&B2_x2Face_dev[ind], Gamma_1);
  Cons1D_to_Prim1D_cu_dev(&Ur_x2Face_dev[ind],&Wr_dev[ind],&B2_x2Face_dev[ind], Gamma_1);

////  printf("i %d j %d\n", j, i);

  flux_roe_cu_dev(Ul_x2Face_dev[ind],Ur_x2Face_dev[ind],Wl_dev[ind],Wr_dev[ind],
          B2_x2Face_dev[ind],x2Flux_dev+ind, Gamma_1, Gamma_2);

//  x2Flux_dev[ind].By = (x2Flux_dev[ind].By/1.0e15)*1.0e15;
//  x2Flux_dev[ind].Bz = (x2Flux_dev[ind].Bz/1.0e15)*1.0e15;
//  x2Flux_dev[ind].E = (x2Flux_dev[ind].E/1.0e15)*1.0e15;
//  x2Flux_dev[ind].Mx = (x2Flux_dev[ind].Mx/1.0e15)*1.0e15;
//  x2Flux_dev[ind].My =(x2Flux_dev[ind].My/1.0e15)*1.0e15;
//  x2Flux_dev[ind].Mz =(x2Flux_dev[ind].Mz/1.0e15)*1.0e15;
//  x2Flux_dev[ind].d =(x2Flux_dev[ind].d/1.0e15)*1.0e15;


////  printf("x2Flux %.15e %.15e %.15e %.15e %.15e %.15e %.15e \n", x2Flux_dev[ind].d, x2Flux_dev[ind].Mx, x2Flux_dev[ind].My, x2Flux_dev[ind].Mz, x2Flux_dev[ind].By, x2Flux_dev[ind].Bz, x2Flux_dev[ind].E);

//  printf("x2Flux %e %e %e %e %e %e %e \n", x2Flux[j][i].d, x2Flux[j][i].Mx, x2Flux[j][i].My, x2Flux[j][i].Mz, x2Flux[j][i].By, x2Flux[j][i].Bz, x2Flux[j][i].E);

//  flux_roe_cu_dev(Ul_x2Face_dev[ind],Ur_x2Face_dev[ind],Wl_dev[i],Wr_dev[i],
//                 B2_x2Face_dev[ind],&x2Flux_dev[ind], Gamma_1, Gamma_2);

//  for (j=js; j<=je+1; j++) {
//      for (i=is-1; i<=ie+1; i++) {
//
//        Cons1D_to_Prim1D(&Ul_x2Face[j][i],&Wl[i],&B2_x2Face[j][i]);
//        Cons1D_to_Prim1D(&Ur_x2Face[j][i],&Wr[i],&B2_x2Face[j][i]);
//
//        flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[i],Wr[i],
//                   B2_x2Face[j][i],&x2Flux[j][i]);
//      }
//    }

}

__global__ void Cons1D_to_Prim1D_Slice1D_8c_next(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B2_x2Face_dev, Cons1D *x2Flux_dev, int is, int ie, int js, int je, int sizex, Real Gamma_1, Real Gamma_2) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = i / sizex;
  i = i % sizex;

  /* Check bounds */
  if(i < is || i > ie || j < js || j > je) return;

  int ind = j*sizex+i;

  /* Main algorithm */
//  Cons1D_to_Prim1D_cu_dev(&Ul_x2Face_dev[ind],&Wl_dev[i],&B2_x2Face_dev[ind], Gamma_1);
//  Cons1D_to_Prim1D_cu_dev(&Ur_x2Face_dev[ind],&Wr_dev[i],&B2_x2Face_dev[ind], Gamma_1);

  flux_roe_cu_dev(Ul_x2Face_dev[ind],Ur_x2Face_dev[ind],Wl_dev[i],Wr_dev[i],
                 B2_x2Face_dev[ind],&x2Flux_dev[ind], Gamma_1, Gamma_2);

//  for (j=js; j<=je+1; j++) {
//      for (i=is-1; i<=ie+1; i++) {
//
//        Cons1D_to_Prim1D(&Ul_x2Face[j][i],&Wl[i],&B2_x2Face[j][i]);
//        Cons1D_to_Prim1D(&Ur_x2Face[j][i],&Wr[i],&B2_x2Face[j][i]);
//
//        flux_roe(Ul_x2Face[j][i],Ur_x2Face[j][i],Wl[i],Wr[i],
//                   B2_x2Face[j][i],&x2Flux[j][i]);
//      }
//    }

}

/**
 * Alternative
 */
__global__ void Cons1D_to_Prim1D_Slice1D_8c_alt(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B2_x2Face_dev, Cons1D *x2Flux_dev, int is, int ie, int j, int sizex, Real Gamma_1) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  /* Check bounds */
  if(i < is || i > ie) return;

  int ind = j*sizex+i;

  /* Main algorithm */
  Cons1D_to_Prim1D_cu_dev(&Ul_x2Face_dev[ind],&Wl_dev[i],&B2_x2Face_dev[ind], Gamma_1);
  Cons1D_to_Prim1D_cu_dev(&Ur_x2Face_dev[ind],&Wr_dev[i],&B2_x2Face_dev[ind], Gamma_1);

}

/**
 * Alternative
 */
__global__ void flux_roe_8c_alt(Cons1D *Ul_x2Face_dev, Cons1D *Ur_x2Face_dev, Prim1D *Wl_dev, Prim1D *Wr_dev, Real *B2_x2Face_dev, Cons1D *x2Flux_dev, int is, int ie, int j, int sizex, Real Gamma_1, Real Gamma_2) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	/* Check bounds */
	if(i < is || i > ie) return;

	int ind = j*sizex+i;

	Cons1D Ul = Ul_x2Face_dev[ind];
	Cons1D Ur = Ur_x2Face_dev[ind];
	Prim1D Wl = Wl_dev[i];
	Prim1D Wr = Wr_dev[i];
	Real Bxi = B2_x2Face_dev[ind];

	Real sqrtdl, sqrtdr, isdlpdr, droe, v1roe, v2roe, v3roe, pbl = 0.0, pbr =
			0.0;
	Real hroe;
	Real b2roe, b3roe, x, y;
	Real coeff[NWAVE];
	Real ev[NWAVE], rem[NWAVE * NWAVE], lem[NWAVE * NWAVE];
	Real dU[NWAVE], a[NWAVE];
#ifdef TEST_INTERMEDIATE_STATES_CU
	Real u_inter[NWAVE],p_inter=0.0;
#endif /* TEST_INTERMEDIATE_STATES */
	/*  Prim1D Wl, Wr; */
	Real *pUl, *pUr, *pFl, *pFr, *pF;
	Cons1D Fl, Fr;
	int n, m, hlle_flag;

	for (n = 0; n < NWAVE; n++) {
		for (m = 0; m < NWAVE; m++) {
			rem[n * NWAVE + m] = 0.0;
			lem[n * NWAVE + m] = 0.0;
		}
	}

	/*--- Step 1. ------------------------------------------------------------------
	 * Convert left- and right- states in conserved to primitive variables.
	 */

	/*
	 pbl = Cons1D_to_Prim1D(&Ul,&Wl,&Bxi);
	 pbr = Cons1D_to_Prim1D(&Ur,&Wr,&Bxi);
	 */

	/*--- Step 2. ------------------------------------------------------------------
	 * Compute Roe-averaged data from left- and right-states
	 */

	sqrtdl = sqrt((Real) Wl.d);
	sqrtdr = sqrt((Real) Wr.d);
	isdlpdr = 1.0 / (sqrtdl + sqrtdr);

	droe = sqrtdl * sqrtdr;
	v1roe = (sqrtdl * Wl.Vx + sqrtdr * Wr.Vx) * isdlpdr;
	v2roe = (sqrtdl * Wl.Vy + sqrtdr * Wr.Vy) * isdlpdr;
	v3roe = (sqrtdl * Wl.Vz + sqrtdr * Wr.Vz) * isdlpdr;

	/* The Roe average of the magnetic field is defined differently  */

	b2roe = (sqrtdr * Wl.By + sqrtdl * Wr.By) * isdlpdr;
	b3roe = (sqrtdr * Wl.Bz + sqrtdl * Wr.Bz) * isdlpdr;
	x = 0.5 * (SQR(Wl.By - Wr.By) + SQR(Wl.Bz - Wr.Bz))
			/ (SQR(sqrtdl + sqrtdr));
	y = 0.5 * (Wl.d + Wr.d) / droe;
	pbl = 0.5 * (SQR(Bxi) + SQR(Wl.By) + SQR(Wl.Bz));
	pbr = 0.5 * (SQR(Bxi) + SQR(Wr.By) + SQR(Wr.Bz));

	/*
	 * Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
	 * rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
	 */

	hroe = ((Ul.E + Wl.P + pbl) / sqrtdl + (Ur.E + Wr.P + pbr) / sqrtdr)
			* isdlpdr;


	/*--- Step 3. ------------------------------------------------------------------
	 * Compute eigenvalues and eigenmatrices using Roe-averaged values
	 */
//	esys_roe_adb_mhd_cu_dev(droe, v1roe, v2roe, v3roe, hroe, Bxi, b2roe, b3roe,
//			x, y, ev, rem, lem, Gamma_1, Gamma_2);

	/*--- Step 4. ------------------------------------------------------------------
	 * Compute L/R fluxes
	 */

	Fl.d = Ul.Mx;
	Fr.d = Ur.Mx;

	Fl.Mx = Ul.Mx * Wl.Vx;
	Fr.Mx = Ur.Mx * Wr.Vx;

	Fl.My = Ul.Mx * Wl.Vy;
	Fr.My = Ur.Mx * Wr.Vy;

	Fl.Mz = Ul.Mx * Wl.Vz;
	Fr.Mz = Ur.Mx * Wr.Vz;

	Fl.Mx += Wl.P;
	Fr.Mx += Wr.P;

	Fl.E = (Ul.E + Wl.P) * Wl.Vx;
	Fr.E = (Ur.E + Wr.P) * Wr.Vx;

	Fl.Mx -= 0.5 * (Bxi * Bxi - SQR(Wl.By) - SQR(Wl.Bz));
	Fr.Mx -= 0.5 * (Bxi * Bxi - SQR(Wr.By) - SQR(Wr.Bz));

	Fl.My -= Bxi * Wl.By;
	Fr.My -= Bxi * Wr.By;

	Fl.Mz -= Bxi * Wl.Bz;
	Fr.Mz -= Bxi * Wr.Bz;

	Fl.E += (pbl * Wl.Vx - Bxi * (Bxi * Wl.Vx + Wl.By * Wl.Vy + Wl.Bz * Wl.Vz));
	Fr.E += (pbr * Wr.Vx - Bxi * (Bxi * Wr.Vx + Wr.By * Wr.Vy + Wr.Bz * Wr.Vz));

	Fl.By = Wl.By * Wl.Vx - Bxi * Wl.Vy;
	Fr.By = Wr.By * Wr.Vx - Bxi * Wr.Vy;

	Fl.Bz = Wl.Bz * Wl.Vx - Bxi * Wl.Vz;
	Fr.Bz = Wr.Bz * Wr.Vx - Bxi * Wr.Vz;

	/*--- Step 5. ------------------------------------------------------------------
	 * Return upwind flux if flow is supersonic
	 */

	if (ev[0] >= 0.0) {
		x2Flux_dev[ind] = Fl;
		return;
	}

	if (ev[NWAVE - 1] <= 0.0) {
		x2Flux_dev[ind] = Fr;
		return;
	}

	/*--- Step 6. ------------------------------------------------------------------
	 * Compute projection of dU onto L eigenvectors ("vector A")
	 */

	pUr = (Real *) &(Ur);
	pUl = (Real *) &(Ul);

	for (n = 0; n < NWAVE; n++)
		dU[n] = pUr[n] - pUl[n];
	for (n = 0; n < NWAVE; n++) {
		a[n] = 0.0;
		for (m = 0; m < NWAVE; m++)
			a[n] += lem[n * NWAVE + m] * dU[m];
	}

	/*--- Step 7. ------------------------------------------------------------------
	 * Check that the density and pressure in the intermediate states are positive.
	 * If not, set hlle_flag=1 if d_inter<0; hlle_flag=2 if p_inter<0, get HLLE
	 * fluxes, and return
	 */

	hlle_flag = 0;
//#ifdef TEST_INTERMEDIATE_STATES_CU

	for (n=0; n<NWAVE; n++) u_inter[n] = pUl[n];
	for (n=0; n<NWAVE-1; n++) {
		for (m=0; m<NWAVE; m++) u_inter[m] = u_inter[m] + __fmul_ru(a[n],rem[m*NWAVE+n]);
		if(ev[n+1] > ev[n]) {
			if (u_inter[0] <= 0.0) {
				hlle_flag=1;
				break;
			}
			p_inter = u_inter[4] - 0.5*(SQR(u_inter[1])+SQR(u_inter[2])+SQR(u_inter[3]))/u_inter[0];
			p_inter -= 0.5*(SQR(u_inter[NWAVE-2])+SQR(u_inter[NWAVE-1])+SQR(Bxi));
			if (p_inter < 0.0) {
				hlle_flag=2;
				break;
			}
		}
	}

	if (hlle_flag != 0) {
		flux_hlle_cu_dev(Ul,Ur,Wl,Wr,Bxi,&x2Flux_dev[ind], Gamma_1, Gamma_2, Gamma_1+1.0);
		return;
	}

//#endif /* TEST_INTERMEDIATE_STATES */

	/*--- Step 8. ------------------------------------------------------------------
	 * Compute Roe flux */

	pFl = (Real *) &(Fl);
	pFr = (Real *) &(Fr);
	pF = (Real *) &(x2Flux_dev[ind]);

	for (m = 0; m < NWAVE; m++) {
		coeff[m] = 0.5 * MAX(fabs(ev[m]), etah_dev) * a[m];
	}
	for (n = 0; n < NWAVE; n++) {
		pF[n] = 0.5 * (pFl[n] + pFr[n]);
		for (m = 0; m < NWAVE; m++) {
			pF[n] -= coeff[m] * rem[n * NWAVE + m];
		}
	}

}
