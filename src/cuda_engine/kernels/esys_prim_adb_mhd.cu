//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

__device__ void esys_prim_adb_mhd_cu_dev(const Real d, const Real v1, const Real p,
  const Real b1, const Real b2, const Real b3,
  Real *eigenvalues,
  Real *rem, Real *lem, Real Gamma)
{
	Real di,cfsq,cf,cssq,cs,bt,bet2,bet3,alpha_f,alpha_s;
	  Real sqrtd,s,a,qf,qs,af,as,vax,na,af_prime,as_prime;
	  Real tsum,tdif,cf2_cs2,ct2;
	  Real btsq,vaxsq,asq;
	  di = 1.0/d;
	  btsq  = b2*b2 + b3*b3;
	  vaxsq = b1*b1*di;
	  asq   = Gamma*p*di;

	/* Compute fast- and slow-magnetosonic speeds (eq. A10) */

	  ct2 = btsq*di;
	  tsum = vaxsq + ct2 + asq;
	  tdif = vaxsq + ct2 - asq;
	  cf2_cs2 = sqrt((Real)(tdif*tdif + 4.0*asq*ct2));

	  cfsq = 0.5*(tsum + cf2_cs2);
	  cf = sqrt((Real)cfsq);

	  cssq = asq*vaxsq/cfsq;
	  cs = sqrt((Real)cssq);

	/* Compute beta(s) (eq A17) */

	  bt  = sqrt(btsq);
	  if (bt == 0.0) {
	    bet2 = 1.0;
	    bet3 = 0.0;
	  } else {
	    bet2 = b2/bt;
	    bet3 = b3/bt;
	  }

	/* Compute alpha(s) (eq A16) */

	  if (cf2_cs2 == 0.0) {
	    alpha_f = 1.0;
	    alpha_s = 0.0;
	  } else if ( (asq - cssq) <= 0.0) {
	    alpha_f = 0.0;
	    alpha_s = 1.0;
	  } else if ( (cfsq - asq) <= 0.0) {
	    alpha_f = 1.0;
	    alpha_s = 0.0;
	  } else {
	    alpha_f = sqrt((asq - cssq)/cf2_cs2);
	    alpha_s = sqrt((cfsq - asq)/cf2_cs2);
	  }

	/* Compute Q(s) and A(s) (eq. A14-15), etc. */

	  sqrtd = sqrt(d);
	  s = SIGN(b1);
	  a = sqrt(asq);
	  qf = cf*alpha_f*s;
	  qs = cs*alpha_s*s;
	  af = a*alpha_f*sqrtd;
	  as = a*alpha_s*sqrtd;

	/* Compute eigenvalues (eq. A9) */

  vax = sqrt(vaxsq);
  eigenvalues[0] = v1 - cf;
  eigenvalues[1] = v1 - vax;
  eigenvalues[2] = v1 - cs;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + cs;
  eigenvalues[5] = v1 + vax;
  eigenvalues[6] = v1 + cf;

/* Right-eigenvectors, stored as COLUMNS (eq. A12) */
/* Note statements are grouped in ROWS for optimization, even though rem[*][n]
 * is the nth right eigenvector */

      rem[0*NWAVE+0] = d*alpha_f;
    /*rem[0*NWAVE+1] = 0.0; */
      rem[0*NWAVE+2] = d*alpha_s;
      rem[0*NWAVE+3] = 1.0;
      rem[0*NWAVE+4] = rem[0*NWAVE+2];
    /*rem[0*NWAVE+5] = 0.0; */
      rem[0*NWAVE+6] = rem[0*NWAVE+0];

      rem[1*NWAVE+0] = -cf*alpha_f;
    /*rem[1*NWAVE+1] = 0.0; */
      rem[1*NWAVE+2] = -cs*alpha_s;
    /*rem[1*NWAVE+3] = 0.0; */
      rem[1*NWAVE+4] = -rem[1*NWAVE+2];
    /*rem[1*NWAVE+5] = 0.0; */
      rem[1*NWAVE+6] = -rem[1*NWAVE+0];

      rem[2*NWAVE+0] = qs*bet2;
      rem[2*NWAVE+1] = -bet3;
      rem[2*NWAVE+2] = -qf*bet2;
    /*rem[2*NWAVE+3] = 0.0; */
      rem[2*NWAVE+4] = -rem[2*NWAVE+2];
      rem[2*NWAVE+5] = bet3;
      rem[2*NWAVE+6] = -rem[2*NWAVE+0];

      rem[3*NWAVE+0] = qs*bet3;
      rem[3*NWAVE+1] = bet2;
      rem[3*NWAVE+2] = -qf*bet3;
    /*rem[3*NWAVE+3] = 0.0; */
      rem[3*NWAVE+4] = -rem[3*NWAVE+2];
      rem[3*NWAVE+5] = -bet2;
      rem[3*NWAVE+6] = -rem[3*NWAVE+0];

      rem[4*NWAVE+0] = d*asq*alpha_f;
    /*rem[4*NWAVE+1] = 0.0; */
      rem[4*NWAVE+2] = d*asq*alpha_s;
    /*rem[4*NWAVE+3] = 0.0; */
      rem[4*NWAVE+4] = rem[4*NWAVE+2];
    /*rem[4*NWAVE+5] = 0.0; */
      rem[4*NWAVE+6] = rem[4*NWAVE+0];

      rem[5*NWAVE+0] = as*bet2;
      rem[5*NWAVE+1] = -bet3*s*sqrtd;
      rem[5*NWAVE+2] = -af*bet2;
    /*rem[5*NWAVE+3] = 0.0; */
      rem[5*NWAVE+4] = rem[5*NWAVE+2];
      rem[5*NWAVE+5] = rem[5*NWAVE+1];
      rem[5*NWAVE+6] = rem[5*NWAVE+0];

      rem[6*NWAVE+0] = as*bet3;
      rem[6*NWAVE+1] = bet2*s*sqrtd;
      rem[6*NWAVE+2] = -af*bet3;
    /*rem[6*NWAVE+3] = 0.0; */
      rem[6*NWAVE+4] = rem[6*NWAVE+2];
      rem[6*NWAVE+5] = rem[6*NWAVE+1];
      rem[6*NWAVE+6] = rem[6*NWAVE+0];

    /* Left-eigenvectors, stored as ROWS (eq. A18) */

      na = 0.5/asq;
      qf = na*qf;
      qs = na*qs;
      af_prime = na*af*di;
      as_prime = na*as*di;

    /*lem[0*NWAVE+0] = 0.0; */
      lem[0*NWAVE+1] = -na*cf*alpha_f;
      lem[0*NWAVE+2] = qs*bet2;
      lem[0*NWAVE+3] = qs*bet3;
      lem[0*NWAVE+4] = na*alpha_f*di;
      lem[0*NWAVE+5] = as_prime*bet2;
      lem[0*NWAVE+6] = as_prime*bet3;

    /*lem[1*NWAVE+0] = 0.0; */
    /*lem[1*NWAVE+1] = 0.0; */
      lem[1*NWAVE+2] = -0.5*bet3;
      lem[1*NWAVE+3] = 0.5*bet2;
    /*lem[1*NWAVE+4] = 0.0; */
      lem[1*NWAVE+5] = -0.5*bet3*s/sqrtd;
      lem[1*NWAVE+6] = 0.5*bet2*s/sqrtd;

    /*lem[2*NWAVE+0] = 0.0; */
      lem[2*NWAVE+1] = -na*cs*alpha_s;
      lem[2*NWAVE+2] = -qf*bet2;
      lem[2*NWAVE+3] = -qf*bet3;
      lem[2*NWAVE+4] = na*alpha_s*di;
      lem[2*NWAVE+5] = -af_prime*bet2;
      lem[2*NWAVE+6] = -af_prime*bet3;

      lem[3*NWAVE+0] = 1.0;
    /*lem[3*NWAVE+1] = 0.0; */
    /*lem[3*NWAVE+2] = 0.0; */
    /*lem[3*NWAVE+3] = 0.0; */
      lem[3*NWAVE+4] = -1.0/asq;
    /*lem[3*NWAVE+5] = 0.0; */
    /*lem[3*NWAVE+6] = 0.0; */

    /*lem[4*NWAVE+0] = 0.0; */
      lem[4*NWAVE+1] = -lem[2*NWAVE+1];
      lem[4*NWAVE+2] = -lem[2*NWAVE+2];
      lem[4*NWAVE+3] = -lem[2*NWAVE+3];
      lem[4*NWAVE+4] = lem[2*NWAVE+4];
      lem[4*NWAVE+5] = lem[2*NWAVE+5];
      lem[4*NWAVE+6] = lem[2*NWAVE+6];

    /*lem[5*NWAVE+0] = 0.0; */
    /*lem[5*NWAVE+1] = 0.0; */
      lem[5*NWAVE+2] = -lem[1*NWAVE+2];
      lem[5*NWAVE+3] = -lem[1*NWAVE+3];
    /*lem[5*NWAVE+4] = 0.0; */
      lem[5*NWAVE+5] = lem[1*NWAVE+5];
      lem[5*NWAVE+6] = lem[1*NWAVE+6];

    /*lem[6*NWAVE+0] = 0.0; */
      lem[6*NWAVE+1] = -lem[0*NWAVE+1];
      lem[6*NWAVE+2] = -lem[0*NWAVE+2];
      lem[6*NWAVE+3] = -lem[0*NWAVE+3];
      lem[6*NWAVE+4] = lem[0*NWAVE+4];
      lem[6*NWAVE+5] = lem[0*NWAVE+5];
      lem[6*NWAVE+6] = lem[0*NWAVE+6];

}
