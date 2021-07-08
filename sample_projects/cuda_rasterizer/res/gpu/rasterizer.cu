// Includes that fix syntax highlighting
#ifdef D3D12_DEBUG
#include "device_launch_parameters.h"
#include "stdio.h"
#include "math_functions.h"
#endif

#include "math_constants.h"

#define gvoid  __global__ void
#define gfloat __global__ float
#define gint   __global__ int

#define dvoid  __device__ void
#define dfloat __device__ float
#define dint   __device__ int

#define cvoid  __constant__ void
#define cfloat __constant__ float
#define cint   __constant__ int

extern "C" {

	cint arrSize;
	gvoid adder(int *arrA, int *arrB, int *result) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, arrSize - 1);
		result[idx] = arrA[idx] + arrB[idx];
	}

	gvoid blank(unsigned char *target, unsigned char *color, int width, int height) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, width * height - 1) * 4;

		target[idx + 0] = color[0];
		target[idx + 1] = color[1];
		target[idx + 2] = color[2];
		target[idx + 3] = color[3];
	}
}
