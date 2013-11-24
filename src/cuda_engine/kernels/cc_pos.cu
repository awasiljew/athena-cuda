//Only for eclipse parsers
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#define __const__
#endif

__device__ void cc_pos_dev(const Grid_gpu *pG, const int i, const int j, Real *px1, Real *px2)
{
  *px1 = pG->x1_0 + ((i + pG->idisp) + 0.5)*pG->dx1;
  *px2 = pG->x2_0 + ((j + pG->jdisp) + 0.5)*pG->dx2;
  return;
}

__device__ void calculateIndexes2D(int *i, int *j, int sizex) {
	*i = (blockIdx.x * blockDim.x) + threadIdx.x;
	*j = *i / sizex;
	*i = *i % sizex;
}

