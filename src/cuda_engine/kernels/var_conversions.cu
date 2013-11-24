//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

__device__ void Prim1D_to_Cons1D_cu_dev(Cons1D *pU, Prim1D *pW, Real *pBx, Real Gamma_1)
{

  pU->d  = pW->d;
  pU->Mx = pW->d*pW->Vx;
  pU->My = pW->d*pW->Vy;
  pU->Mz = pW->d*pW->Vz;
  pU->E = pW->P/Gamma_1 + 0.5*pW->d*(SQR(pW->Vx) + SQR(pW->Vy) + SQR(pW->Vz));
  pU->E += 0.5*(SQR(*pBx) + SQR(pW->By) + SQR(pW->Bz));
  pU->By = pW->By;
  pU->Bz = pW->Bz;

}

__device__ void Cons1D_to_Prim1D_cu_dev(Cons1D *pU, Prim1D *pW, Real *pBx, Real Gamma_1) {
	Real di = 1.0 / pU->d;

	pW->d = pU->d;
	pW->Vx = pU->Mx * di;
	pW->Vy = pU->My * di;
	pW->Vz = pU->Mz * di;

	pW->P = pU->E - 0.5 * (SQR(pU->Mx) + SQR(pU->My) + SQR(pU->Mz)) * di;
	pW->P -= 0.5 * (SQR(*pBx) + SQR(pU->By) + SQR(pU->Bz));
	pW->P *= Gamma_1;
	pW->P = MAX(pW->P, TINY_NUMBER);

	pW->By = pU->By;
	pW->Bz = pU->Bz;

	return;
}
