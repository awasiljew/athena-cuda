//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

//////////// CUDA //////////////////////
/*----------------------------------------------------------------------------*/
/* lr_states:
 * Input Arguments:
 *   W = PRIMITIVE variables at cell centers along 1-D slice
 *   Bxc = B in direction of slice at cell center
 *   dt = timestep;   dtodx = dt/dx
 *   il,iu = lower and upper indices of zone centers in slice
 * W and Bxc must be initialized over [il-2:iu+2]
 *
 * Output Arguments:
 *   Wl,Wr = L/R-states of PRIMITIVE variables at interfaces over [il:iu+1]
 */
// Started as:
// lr_states_cu<<<nBlocks, BLOCK_SIZE>>>(W_dev, Bxc_dev, pG->dt, dtodx1, is-2, ie+2, Wl_dev, Wr_dev, Gamma);
// lr_states_cu<<<nBlocks_y, BLOCK_SIZE>>>(W_dev, Bxc_dev, pG->dt, dtodx2, js-1, je+1, Wl_dev, Wr_dev, /*pW_dev, */Gamma); ///was js-1 and je+1
// Original: lr_states(W,Bxc,pG->dt,dtodx1,is-1,ie+1,Wl,Wr);
__global__ void lr_states_cu(Prim1D* W, Real* Bxc,
               Real dt, Real dtodx, int is, int ie,
               Prim1D* Wl, Prim1D* Wr, Real Gamma) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // is == is-1
  // ie == ie+1
  // So the final loop will be from is-1 .. ie+1, but should be from is-2 .. ie+2
  if((i < is) || (i > ie)) return;

  int n,m;
  Real lim_slope1,lim_slope2,qa,qx;
  Real ev[NWAVE],rem[NWAVE*NWAVE],lem[NWAVE*NWAVE];
  Real dWc[NWAVE],dWl[NWAVE];
  Real dWr[NWAVE],dWg[NWAVE];
  Real dac[NWAVE],dal[NWAVE];
  Real dar[NWAVE],dag[NWAVE],da[NWAVE];
  Real Wlv[NWAVE],Wrv[NWAVE];
  Real dW[NWAVE],dWm[NWAVE];
  Real *pWl, *pWr;

/* Zero eigenmatrices, set pointer to primitive variables */
  for (n=0; n<NWAVE; n++) {
    for (m=0; m<NWAVE; m++) {
      rem[n*NWAVE+m] = 0.0;
      lem[n*NWAVE+m] = 0.0;
    }
  }

  Real* pW_dev = (Real*)&(W[i]); //W[i]
  Real* pW_dev_1 = (Real*)&(W[i-1]); //W[i-1]
  Real* pW_dev_2 = (Real*)&(W[i+1]); //W[i+1]

/*========================== MAIN CALCULATION =======================*/

/*--- Step 1. ------------------------------------------------------------------
 * Compute eigensystem in primitive variables.  */

  esys_prim_adb_mhd_cu_dev(W[i].d, W[i].Vx, W[i].P, Bxc[i], W[i].By, W[i].Bz, ev, lem, rem, Gamma);

////////////////////////////////////////////////////////////////////////////////
//
//  Real d = W[i].d;
//  Real v1 = W[i].Vx;
//  Real p = W[i].P;
//  Real b1 = Bxc[i];
//  Real b2 = W[i].By;
//  Real b3 = W[i].Bz;
//
//  Real di,cfsq,cf,cssq,cs,bt,bet2,bet3,alpha_f,alpha_s;
//  Real sqrtd,s,a,qf,qs,af,as,vax,na,af_prime,as_prime;
//  Real tsum,tdif,cf2_cs2,ct2;
//  Real btsq,vaxsq,asq;
//  di = 1.0/d;
//  btsq  = b2*b2 + b3*b3;
//  vaxsq = b1*b1*di;
//  asq   = Gamma*p*di;
//
//  /* Compute fast- and slow-magnetosonic speeds (eq. A10) */
//
//    ct2 = btsq*di;
//    tsum = vaxsq + ct2 + asq;
//    tdif = vaxsq + ct2 - asq;
//    cf2_cs2 = sqrt((Real)(tdif*tdif + 4.0*asq*ct2));
//
//    cfsq = 0.5*(tsum + cf2_cs2);
//    cf = sqrt((Real)cfsq);
//
//    cssq = asq*vaxsq/cfsq;
//    cs = sqrt((Real)cssq);
//
//  /* Compute beta(s) (eq A17) */
//
//    bt  = sqrt(btsq);
//    if (bt == 0.0) {
//      bet2 = 1.0;
//      bet3 = 0.0;
//    } else {
//      bet2 = b2/bt;
//      bet3 = b3/bt;
//    }
//
//  /* Compute alpha(s) (eq A16) */
//
//    if (cf2_cs2 == 0.0) {
//      alpha_f = 1.0;
//      alpha_s = 0.0;
//    } else if ( (asq - cssq) <= 0.0) {
//      alpha_f = 0.0;
//      alpha_s = 1.0;
//    } else if ( (cfsq - asq) <= 0.0) {
//      alpha_f = 1.0;
//      alpha_s = 0.0;
//    } else {
//      alpha_f = sqrt((asq - cssq)/cf2_cs2);
//      alpha_s = sqrt((cfsq - asq)/cf2_cs2);
//    }
//
//  /* Compute Q(s) and A(s) (eq. A14-15), etc. */
//
//    sqrtd = sqrt(d);
//    s = SIGN(b1);
//    a = sqrt(asq);
//    qf = cf*alpha_f*s;
//    qs = cs*alpha_s*s;
//    af = a*alpha_f*sqrtd;
//    as = a*alpha_s*sqrtd;
//
//  /* Compute eigenvalues (eq. A9) */
//
//    vax = sqrt(vaxsq);
//    ev[0] = v1 - cf;
//    ev[1] = v1 - vax;
//    ev[2] = v1 - cs;
//    ev[3] = v1;
//    ev[4] = v1 + cs;
//    ev[5] = v1 + vax;
//    ev[6] = v1 + cf;
//
//  /* Right-eigenvectors, stored as COLUMNS (eq. A12) */
//  /* Note statements are grouped in ROWS for optimization, even though rem[*][n]
//   * is the nth right eigenvector */
//
//
//    rem[0*NWAVE+0] = d*alpha_f;
//  /*rem[0*NWAVE+1] = 0.0; */
//    rem[0*NWAVE+2] = d*alpha_s;
//    rem[0*NWAVE+3] = 1.0;
//    rem[0*NWAVE+4] = rem[0*NWAVE+2];
//  /*rem[0*NWAVE+5] = 0.0; */
//    rem[0*NWAVE+6] = rem[0*NWAVE+0];
//
//    rem[1*NWAVE+0] = -cf*alpha_f;
//  /*rem[1*NWAVE+1] = 0.0; */
//    rem[1*NWAVE+2] = -cs*alpha_s;
//  /*rem[1*NWAVE+3] = 0.0; */
//    rem[1*NWAVE+4] = -rem[1*NWAVE+2];
//  /*rem[1*NWAVE+5] = 0.0; */
//    rem[1*NWAVE+6] = -rem[1*NWAVE+0];
//
//    rem[2*NWAVE+0] = qs*bet2;
//    rem[2*NWAVE+1] = -bet3;
//    rem[2*NWAVE+2] = -qf*bet2;
//  /*rem[2*NWAVE+3] = 0.0; */
//    rem[2*NWAVE+4] = -rem[2*NWAVE+2];
//    rem[2*NWAVE+5] = bet3;
//    rem[2*NWAVE+6] = -rem[2*NWAVE+0];
//
//    rem[3*NWAVE+0] = qs*bet3;
//    rem[3*NWAVE+1] = bet2;
//    rem[3*NWAVE+2] = -qf*bet3;
//  /*rem[3*NWAVE+3] = 0.0; */
//    rem[3*NWAVE+4] = -rem[3*NWAVE+2];
//    rem[3*NWAVE+5] = -bet2;
//    rem[3*NWAVE+6] = -rem[3*NWAVE+0];
//
//    rem[4*NWAVE+0] = d*asq*alpha_f;
//  /*rem[4*NWAVE+1] = 0.0; */
//    rem[4*NWAVE+2] = d*asq*alpha_s;
//  /*rem[4*NWAVE+3] = 0.0; */
//    rem[4*NWAVE+4] = rem[4*NWAVE+2];
//  /*rem[4*NWAVE+5] = 0.0; */
//    rem[4*NWAVE+6] = rem[4*NWAVE+0];
//
//    rem[5*NWAVE+0] = as*bet2;
//    rem[5*NWAVE+1] = -bet3*s*sqrtd;
//    rem[5*NWAVE+2] = -af*bet2;
//  /*rem[5*NWAVE+3] = 0.0; */
//    rem[5*NWAVE+4] = rem[5*NWAVE+2];
//    rem[5*NWAVE+5] = rem[5*NWAVE+1];
//    rem[5*NWAVE+6] = rem[5*NWAVE+0];
//
//    rem[6*NWAVE+0] = as*bet3;
//    rem[6*NWAVE+1] = bet2*s*sqrtd;
//    rem[6*NWAVE+2] = -af*bet3;
//  /*rem[6*NWAVE+3] = 0.0; */
//    rem[6*NWAVE+4] = rem[6*NWAVE+2];
//    rem[6*NWAVE+5] = rem[6*NWAVE+1];
//    rem[6*NWAVE+6] = rem[6*NWAVE+0];
//
//  /* Left-eigenvectors, stored as ROWS (eq. A18) */
//
//    na = 0.5/asq;
//    qf = na*qf;
//    qs = na*qs;
//    af_prime = na*af*di;
//    as_prime = na*as*di;
//
//  /*lem[0*NWAVE+0] = 0.0; */
//    lem[0*NWAVE+1] = -na*cf*alpha_f;
//    lem[0*NWAVE+2] = qs*bet2;
//    lem[0*NWAVE+3] = qs*bet3;
//    lem[0*NWAVE+4] = na*alpha_f*di;
//    lem[0*NWAVE+5] = as_prime*bet2;
//    lem[0*NWAVE+6] = as_prime*bet3;
//
//  /*lem[1*NWAVE+0] = 0.0; */
//  /*lem[1*NWAVE+1] = 0.0; */
//    lem[1*NWAVE+2] = -0.5*bet3;
//    lem[1*NWAVE+3] = 0.5*bet2;
//  /*lem[1*NWAVE+4] = 0.0; */
//    lem[1*NWAVE+5] = -0.5*bet3*s/sqrtd;
//    lem[1*NWAVE+6] = 0.5*bet2*s/sqrtd;
//
//  /*lem[2*NWAVE+0] = 0.0; */
//    lem[2*NWAVE+1] = -na*cs*alpha_s;
//    lem[2*NWAVE+2] = -qf*bet2;
//    lem[2*NWAVE+3] = -qf*bet3;
//    lem[2*NWAVE+4] = na*alpha_s*di;
//    lem[2*NWAVE+5] = -af_prime*bet2;
//    lem[2*NWAVE+6] = -af_prime*bet3;
//
//    lem[3*NWAVE+0] = 1.0;
//  /*lem[3*NWAVE+1] = 0.0; */
//  /*lem[3*NWAVE+2] = 0.0; */
//  /*lem[3*NWAVE+3] = 0.0; */
//    lem[3*NWAVE+4] = -1.0/asq;
//  /*lem[3*NWAVE+5] = 0.0; */
//  /*lem[3*NWAVE+6] = 0.0; */
//
//  /*lem[4*NWAVE+0] = 0.0; */
//    lem[4*NWAVE+1] = -lem[2*NWAVE+1];
//    lem[4*NWAVE+2] = -lem[2*NWAVE+2];
//    lem[4*NWAVE+3] = -lem[2*NWAVE+3];
//    lem[4*NWAVE+4] = lem[2*NWAVE+4];
//    lem[4*NWAVE+5] = lem[2*NWAVE+5];
//    lem[4*NWAVE+6] = lem[2*NWAVE+6];
//
//  /*lem[5*NWAVE+0] = 0.0; */
//  /*lem[5*NWAVE+1] = 0.0; */
//    lem[5*NWAVE+2] = -lem[1*NWAVE+2];
//    lem[5*NWAVE+3] = -lem[1*NWAVE+3];
//  /*lem[5*NWAVE+4] = 0.0; */
//    lem[5*NWAVE+5] = lem[1*NWAVE+5];
//    lem[5*NWAVE+6] = lem[1*NWAVE+6];
//
//  /*lem[6*NWAVE+0] = 0.0; */
//    lem[6*NWAVE+1] = -lem[0*NWAVE+1];
//    lem[6*NWAVE+2] = -lem[0*NWAVE+2];
//    lem[6*NWAVE+3] = -lem[0*NWAVE+3];
//    lem[6*NWAVE+4] = lem[0*NWAVE+4];
//    lem[6*NWAVE+5] = lem[0*NWAVE+5];
//    lem[6*NWAVE+6] = lem[0*NWAVE+6];
//
////////////////////////////////////////////////////////////////////////////////

/*--- Step 2. ------------------------------------------------------------------
 * Compute centered, L/R, and van Leer differences of primitive variables
 * Note we access contiguous array elements by indexing pointers for speed */

  for (n=0; n<NWAVE; n++) {
    dWc[n] = pW_dev_2[n] - pW_dev_1[n];
    dWl[n] = pW_dev[n]   - pW_dev_1[n];
    dWr[n] = pW_dev_2[n] - pW_dev[n];
    if (dWl[n]*dWr[n] > 0.0) {
      dWg[n] = 2.0*dWl[n]*dWr[n]/(dWl[n]+dWr[n]);
    } else {
      dWg[n] = 0.0;
    }
  }


/*--- Step 3. ------------------------------------------------------------------
 * Project differences in primitive variables along characteristics */

  for (n=0; n<NWAVE; n++) {
    dac[n] = lem[n*NWAVE+0]*dWc[0];
    dal[n] = lem[n*NWAVE+0]*dWl[0];
    dar[n] = lem[n*NWAVE+0]*dWr[0];
    dag[n] = lem[n*NWAVE+0]*dWg[0];
    for (m=1; m<NWAVE; m++) {
      dac[n] += lem[n*NWAVE+m]*dWc[m];
      dal[n] += lem[n*NWAVE+m]*dWl[m];
      dar[n] += lem[n*NWAVE+m]*dWr[m];
      dag[n] += lem[n*NWAVE+m]*dWg[m];
    }
  }

/*--- Step 4. ------------------------------------------------------------------
 * Apply monotonicity constraints to characteristic projections */

  for (n=0; n<NWAVE; n++) {
    da[n] = 0.0;
    if (dal[n]*dar[n] > 0.0) {
      lim_slope1 = MIN(    fabs(dal[n]),fabs(dar[n]));
      lim_slope2 = MIN(0.5*fabs(dac[n]),fabs(dag[n]));
      da[n] = SIGN(dac[n])*MIN(2.0*lim_slope1,lim_slope2);
    }
  }

/*--- Step 5. ------------------------------------------------------------------
 * Project monotonic slopes in characteristic back to primitive variables  */

  for (n=0; n<NWAVE; n++) {
    dWm[n] = da[0]*rem[n*NWAVE+0];
    for (m=1; m<NWAVE; m++) {
      dWm[n] += da[m]*rem[n*NWAVE+m];
    }
  }

/*--- Step 7. ------------------------------------------------------------------
 * Compute L/R values, ensure they lie between neighboring cell-centered vals */

  for (n=0; n<NWAVE; n++) {
    Wlv[n] = pW_dev[n] - 0.5*dWm[n];
    Wrv[n] = pW_dev[n] + 0.5*dWm[n];
  }

  for (n=0; n<NWAVE; n++) {
    Wlv[n] = MAX(MIN(pW_dev[n],pW_dev_1[n]),Wlv[n]);
    Wlv[n] = MIN(MAX(pW_dev[n],pW_dev_1[n]),Wlv[n]);
    Wrv[n] = MAX(MIN(pW_dev[n],pW_dev_2[n]),Wrv[n]);
    Wrv[n] = MIN(MAX(pW_dev[n],pW_dev_2[n]),Wrv[n]);
  }

  for (n=0; n<NWAVE; n++) {
    dW[n] = Wrv[n] - Wlv[n];
  }

/*--- Step 8. ------------------------------------------------------------------
 * Integrate linear interpolation function over domain of dependence defined by
 * max(min) eigenvalue
 */

  pWl = (Real *) &(Wl[i+1]);
  pWr = (Real *) &(Wr[i]);

  qx = 0.5*MAX(ev[NWAVE-1],0.0)*dtodx;

  for (n=0; n<NWAVE; n++) {
    pWl[n] = Wrv[n] - qx*dW[n];
  }

  qx = -0.5*MIN(ev[0],0.0)*dtodx;
  for (n=0; n<NWAVE; n++) {
    pWr[n] = Wlv[n] + qx*dW[n];
  }

/*--- Step 9. ------------------------------------------------------------------
 * Then subtract amount of each wave n that does not reach the interface
 * during timestep (CW eqn 3.5ff).  For HLL fluxes, must subtract waves that
 * move in both directions.
 */

  for (n=0; n<NWAVE; n++) {
    if (ev[n] > 0.0) {
      qa  = 0.0;
      for (m=0; m<NWAVE; m++) {
        qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[NWAVE-1]-ev[n])*dW[m];
      }
      for (m=0; m<NWAVE; m++) pWl[m] += qa*rem[m*NWAVE+n];
    }
  }

  for (n=0; n<NWAVE; n++) {
    if (ev[n] < 0.) {
      qa = 0.0;
      for (m=0; m<NWAVE; m++) {
        qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[0]-ev[n])*dW[m];
      }
      for (m=0; m<NWAVE; m++) pWr[m] += qa*rem[m*NWAVE+n];
        qa  = 0.0;
        for (m=0; m<NWAVE; m++) {
          qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[n]-ev[NWAVE-1])*dW[m];
        }
        for (m=0; m<NWAVE; m++) pWl[m] -= qa*rem[m*NWAVE+n];
    }
  }

/* Wave subtraction for advected quantities */
  for (n=NWAVE; n<NWAVE; n++) {
    if (W[i].Vx > 0.) {
      pWl[n] += 0.5*dtodx*(ev[NWAVE-1]-W[i].Vx)*dW[n];
    } else if (W[i].Vx < 0.) {
      pWr[n] += 0.5*dtodx*(ev[0]-W[i].Vx)*dW[n];
    }
  }

/*===================== END MAIN ALGORITHM ===========================*/

//#ifdef __DEVICE_EMULATION__
//
//  /* Check */
//  //if(Wl_shared[threadIdx.x].By != 0.0f && fabs(Wl_shared[threadIdx.x].By) < 1.0e-22 ) {
////	  printf("BY LESS THAN TINY NUMBER!!! %d %e\n", i+1, Wl_shared[threadIdx.x].By);
//  //}
//
//#endif

}


__global__ void lr_states_cu_1b_dev(Prim1D* W, Real* Bxc,
               Real dt, Real dtodx, int is, int ie, int j, int sizex,
               Prim1D* Wl, Prim1D* Wr, Real Gamma) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // is == is-1
  // ie == ie+1
  // So the final loop will be from is-1 .. ie+1, but should be from is-2 .. ie+2
  if((i < is) || (i > ie)) return;

  i = j*sizex+i;

  int n,m;
  Real lim_slope1,lim_slope2,qa,qx;
  Real ev[NWAVE],rem[NWAVE*NWAVE],lem[NWAVE*NWAVE];
  Real dWc[NWAVE],dWl[NWAVE];
  Real dWr[NWAVE],dWg[NWAVE];
  Real dac[NWAVE],dal[NWAVE];
  Real dar[NWAVE],dag[NWAVE],da[NWAVE];
  Real Wlv[NWAVE],Wrv[NWAVE];
  Real dW[NWAVE],dWm[NWAVE];
  Real *pWl, *pWr;

/* Zero eigenmatrices, set pointer to primitive variables */
  for (n=0; n<NWAVE; n++) {
    for (m=0; m<NWAVE; m++) {
      rem[n*NWAVE+m] = 0.0;
      lem[n*NWAVE+m] = 0.0;
    }
  }

  Real* pW_dev = (Real*)&(W[i]); //W[i]
  Real* pW_dev_1 = (Real*)&(W[i-1]); //W[i-1]
  Real* pW_dev_2 = (Real*)&(W[i+1]); //W[i+1]

/*========================== MAIN CALCULATION =======================*/

  /*--- Step 1. ------------------------------------------------------------------
   * Compute eigensystem in primitive variables.  */

      esys_prim_adb_mhd_cu_dev(W[i].d,W[i].Vx,W[i].P,Bxc[i],W[i].By,W[i].Bz,ev,rem,lem, Gamma);

  /*--- Step 2. ------------------------------------------------------------------
   * Compute centered, L/R, and van Leer differences of primitive variables
   * Note we access contiguous array elements by indexing pointers for speed */

      for (n=0; n<NWAVE; n++) {
        dWc[n] = pW_dev_2[n] - pW_dev_1[n];
        dWl[n] = pW_dev[n]   - pW_dev_1[n];
        dWr[n] = pW_dev_2[n] - pW_dev[n];
        if (dWl[n]*dWr[n] > 0.0) {
          dWg[n] = 2.0*dWl[n]*dWr[n]/(dWl[n]+dWr[n]);
        } else {
          dWg[n] = 0.0;
        }
      }

  /*--- Step 3. ------------------------------------------------------------------
   * Project differences in primitive variables along characteristics */

      for (n=0; n<NWAVE; n++) {
        dac[n] = lem[n*NWAVE+0]*dWc[0];
        dal[n] = lem[n*NWAVE+0]*dWl[0];
        dar[n] = lem[n*NWAVE+0]*dWr[0];
        dag[n] = lem[n*NWAVE+0]*dWg[0];
        for (m=1; m<NWAVE; m++) {
  	dac[n] += lem[n*NWAVE+m]*dWc[m];
  	dal[n] += lem[n*NWAVE+m]*dWl[m];
  	dar[n] += lem[n*NWAVE+m]*dWr[m];
  	dag[n] += lem[n*NWAVE+m]*dWg[m];
        }
      }

  /*--- Step 4. ------------------------------------------------------------------
   * Apply monotonicity constraints to characteristic projections */

      for (n=0; n<NWAVE; n++) {
        da[n] = 0.0;
        if (dal[n]*dar[n] > 0.0) {
          lim_slope1 = MIN(    fabs(dal[n]),fabs(dar[n]));
          lim_slope2 = MIN(0.5*fabs(dac[n]),fabs(dag[n]));
          da[n] = SIGN(dac[n])*MIN(2.0*lim_slope1,lim_slope2);
        }
      }

  /*--- Step 5. ------------------------------------------------------------------
   * Project monotonic slopes in characteristic back to primitive variables  */

      for (n=0; n<NWAVE; n++) {
        dWm[n] = da[0]*rem[n*NWAVE+0];
        for (m=1; m<NWAVE; m++) {
          dWm[n] += da[m]*rem[n*NWAVE+m];
        }
      }

  /*--- Step 7. ------------------------------------------------------------------
   * Compute L/R values, ensure they lie between neighboring cell-centered vals */

      for (n=0; n<NWAVE; n++) {
        Wlv[n] = pW_dev[n] - 0.5*dWm[n];
        Wrv[n] = pW_dev[n] + 0.5*dWm[n];
      }

      for (n=0; n<NWAVE; n++) {
        Wlv[n] = MAX(MIN(pW_dev[n],pW_dev_1[n]),Wlv[n]);
        Wlv[n] = MIN(MAX(pW_dev[n],pW_dev_1[n]),Wlv[n]);
        Wrv[n] = MAX(MIN(pW_dev[n],pW_dev_2[n]),Wrv[n]);
        Wrv[n] = MIN(MAX(pW_dev[n],pW_dev_2[n]),Wrv[n]);
      }

      for (n=0; n<NWAVE; n++) {
        dW[n] = Wrv[n] - Wlv[n];
      }

  /*--- Step 8. ------------------------------------------------------------------
   * Integrate linear interpolation function over domain of dependence defined by
   * max(min) eigenvalue
   */

      pWl = (Real *) &(Wl[i+1]);
      pWr = (Real *) &(Wr[i]);

      qx = 0.5*MAX(ev[NWAVE-1],0.0)*dtodx;
      for (n=0; n<NWAVE; n++) {
        pWl[n] = Wrv[n] - qx*dW[n];
      }

      qx = -0.5*MIN(ev[0],0.0)*dtodx;
      for (n=0; n<NWAVE; n++) {
        pWr[n] = Wlv[n] + qx*dW[n];
      }


  /*--- Step 9. ------------------------------------------------------------------
   * Then subtract amount of each wave n that does not reach the interface
   * during timestep (CW eqn 3.5ff).  For HLL fluxes, must subtract waves that
   * move in both directions.
   */

      for (n=0; n<NWAVE; n++) {
        if (ev[n] > 0.) {
  	qa  = 0.0;
  	for (m=0; m<NWAVE; m++) {
  	  qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[NWAVE-1]-ev[n])*dW[m];
  	}
  	for (m=0; m<NWAVE; m++) pWl[m] += qa*rem[m*NWAVE+n];
        }
      }

      for (n=0; n<NWAVE; n++) {
        if (ev[n] < 0.) {
          qa = 0.0;
          for (m=0; m<NWAVE; m++) {
            qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[0]-ev[n])*dW[m];
          }
          for (m=0; m<NWAVE; m++) pWr[m] += qa*rem[m*NWAVE+n];
  	qa  = 0.0;
  	for (m=0; m<NWAVE; m++) {
  	  qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[n]-ev[NWAVE-1])*dW[m];
  	}
  	for (m=0; m<NWAVE; m++) pWl[m] -= qa*rem[m*NWAVE+n];
        }
      }

  /* Wave subtraction for advected quantities */
      for (n=NWAVE; n<NWAVE; n++) {
        if (W[i].Vx > 0.) {
          pWl[n] += 0.5*dtodx*(ev[NWAVE-1]-W[i].Vx)*dW[n];
        } else if (W[i].Vx < 0.) {
          pWr[n] += 0.5*dtodx*(ev[0]-W[i].Vx)*dW[n];
        }
      }

/*===================== END MAIN ALGORITHM ===========================*/

}


__global__ void lr_states_cu_2b_dev(Prim1D* W, Real* Bxc,
               Real dt, Real dtodx, int js, int je, int i, int sizex, int sizey,
               Prim1D* Wl, Prim1D* Wr, Real Gamma) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // is == is-1
  // ie == ie+1
  // So the final loop will be from is-1 .. ie+1, but should be from is-2 .. ie+2
  if((j < js) || (j > je)) return;

  j = i*sizex+j;

  int n,m;
    Real lim_slope1,lim_slope2,qa,qx;
    Real ev[NWAVE],rem[NWAVE*NWAVE],lem[NWAVE*NWAVE];
    Real dWc[NWAVE],dWl[NWAVE];
    Real dWr[NWAVE],dWg[NWAVE];
    Real dac[NWAVE],dal[NWAVE];
    Real dar[NWAVE],dag[NWAVE],da[NWAVE];
    Real Wlv[NWAVE],Wrv[NWAVE];
    Real dW[NWAVE],dWm[NWAVE];
    Real *pWl, *pWr;

  /* Zero eigenmatrices, set pointer to primitive variables */
    for (n=0; n<NWAVE; n++) {
      for (m=0; m<NWAVE; m++) {
        rem[n*NWAVE+m] = 0.0;
        lem[n*NWAVE+m] = 0.0;
      }
    }

    Real* pW_dev = (Real*)&(W[j]); //W[i]
    Real* pW_dev_1 = (Real*)&(W[j-1]); //W[i-1]
    Real* pW_dev_2 = (Real*)&(W[j+1]); //W[i+1]

  /*========================== MAIN CALCULATION =======================*/

    /*--- Step 1. ------------------------------------------------------------------
     * Compute eigensystem in primitive variables.  */

        esys_prim_adb_mhd_cu_dev(W[j].d,W[j].Vx,W[j].P,Bxc[j],W[j].By,W[j].Bz,ev,rem,lem, Gamma);

    /*--- Step 2. ------------------------------------------------------------------
     * Compute centered, L/R, and van Leer differences of primitive variables
     * Note we access contiguous array elements by indexing pointers for speed */

        for (n=0; n<NWAVE; n++) {
          dWc[n] = pW_dev_2[n] - pW_dev_1[n];
          dWl[n] = pW_dev[n]   - pW_dev_1[n];
          dWr[n] = pW_dev_2[n] - pW_dev[n];
          if (dWl[n]*dWr[n] > 0.0) {
            dWg[n] = 2.0*dWl[n]*dWr[n]/(dWl[n]+dWr[n]);
          } else {
            dWg[n] = 0.0;
          }
        }

    /*--- Step 3. ------------------------------------------------------------------
     * Project differences in primitive variables along characteristics */

        for (n=0; n<NWAVE; n++) {
          dac[n] = lem[n*NWAVE+0]*dWc[0];
          dal[n] = lem[n*NWAVE+0]*dWl[0];
          dar[n] = lem[n*NWAVE+0]*dWr[0];
          dag[n] = lem[n*NWAVE+0]*dWg[0];
          for (m=1; m<NWAVE; m++) {
    	dac[n] += lem[n*NWAVE+m]*dWc[m];
    	dal[n] += lem[n*NWAVE+m]*dWl[m];
    	dar[n] += lem[n*NWAVE+m]*dWr[m];
    	dag[n] += lem[n*NWAVE+m]*dWg[m];
          }
        }

    /*--- Step 4. ------------------------------------------------------------------
     * Apply monotonicity constraints to characteristic projections */

        for (n=0; n<NWAVE; n++) {
          da[n] = 0.0;
          if (dal[n]*dar[n] > 0.0) {
            lim_slope1 = MIN(    fabs(dal[n]),fabs(dar[n]));
            lim_slope2 = MIN(0.5*fabs(dac[n]),fabs(dag[n]));
            da[n] = SIGN(dac[n])*MIN(2.0*lim_slope1,lim_slope2);
          }
        }

    /*--- Step 5. ------------------------------------------------------------------
     * Project monotonic slopes in characteristic back to primitive variables  */

        for (n=0; n<NWAVE; n++) {
          dWm[n] = da[0]*rem[n*NWAVE+0];
          for (m=1; m<NWAVE; m++) {
            dWm[n] += da[m]*rem[n*NWAVE+m];
          }
        }

    /*--- Step 7. ------------------------------------------------------------------
     * Compute L/R values, ensure they lie between neighboring cell-centered vals */

        for (n=0; n<NWAVE; n++) {
          Wlv[n] = pW_dev[n] - 0.5*dWm[n];
          Wrv[n] = pW_dev[n] + 0.5*dWm[n];
        }

        for (n=0; n<NWAVE; n++) {
          Wlv[n] = MAX(MIN(pW_dev[n],pW_dev_1[n]),Wlv[n]);
          Wlv[n] = MIN(MAX(pW_dev[n],pW_dev_1[n]),Wlv[n]);
          Wrv[n] = MAX(MIN(pW_dev[n],pW_dev_2[n]),Wrv[n]);
          Wrv[n] = MIN(MAX(pW_dev[n],pW_dev_2[n]),Wrv[n]);
        }

        for (n=0; n<NWAVE; n++) {
          dW[n] = Wrv[n] - Wlv[n];
        }

    /*--- Step 8. ------------------------------------------------------------------
     * Integrate linear interpolation function over domain of dependence defined by
     * max(min) eigenvalue
     */

        pWl = (Real *) &(Wl[j+1]);
        pWr = (Real *) &(Wr[j]);

        qx = 0.5*MAX(ev[NWAVE-1],0.0)*dtodx;
        for (n=0; n<NWAVE; n++) {
          pWl[n] = Wrv[n] - qx*dW[n];
        }

        qx = -0.5*MIN(ev[0],0.0)*dtodx;
        for (n=0; n<NWAVE; n++) {
          pWr[n] = Wlv[n] + qx*dW[n];
        }


    /*--- Step 9. ------------------------------------------------------------------
     * Then subtract amount of each wave n that does not reach the interface
     * during timestep (CW eqn 3.5ff).  For HLL fluxes, must subtract waves that
     * move in both directions.
     */

        for (n=0; n<NWAVE; n++) {
          if (ev[n] > 0.) {
    	qa  = 0.0;
    	for (m=0; m<NWAVE; m++) {
    	  qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[NWAVE-1]-ev[n])*dW[m];
    	}
    	for (m=0; m<NWAVE; m++) pWl[m] += qa*rem[m*NWAVE+n];
          }
        }

        for (n=0; n<NWAVE; n++) {
          if (ev[n] < 0.) {
            qa = 0.0;
            for (m=0; m<NWAVE; m++) {
              qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[0]-ev[n])*dW[m];
            }
            for (m=0; m<NWAVE; m++) pWr[m] += qa*rem[m*NWAVE+n];
    	qa  = 0.0;
    	for (m=0; m<NWAVE; m++) {
    	  qa += lem[n*NWAVE+m]*0.5*dtodx*(ev[n]-ev[NWAVE-1])*dW[m];
    	}
    	for (m=0; m<NWAVE; m++) pWl[m] -= qa*rem[m*NWAVE+n];
          }
        }

    /* Wave subtraction for advected quantities */
        for (n=NWAVE; n<NWAVE; n++) {
          if (W[j].Vx > 0.) {
            pWl[n] += 0.5*dtodx*(ev[NWAVE-1]-W[j].Vx)*dW[n];
          } else if (W[j].Vx < 0.) {
            pWr[n] += 0.5*dtodx*(ev[0]-W[j].Vx)*dW[n];
          }
        }

  /*===================== END MAIN ALGORITHM ===========================*/
}
