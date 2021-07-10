// Includes that fix syntax highlighting
#ifdef D3D12_DEBUG
#include "device_launch_parameters.h"
#include "stdio.h"
#include "math_functions.h"
#endif

#include "math_constants.h"

#include "cuda_cpu_common.h"
#define gvoid   __global__ void
#define gint    __global__ int
#define guint   __global__ unsigned int
#define gbool   __global__ bool
#define gfloat  __global__ float
#define gfloat2 __global__ float2
#define gfloat3 __global__ float3
#define gfloat4 __global__ float4

#define dvoid   __device__ void
#define dint    __device__ int
#define duint   __device__ unsigned int
#define dbool   __device__ bool
#define dfloat  __device__ float
#define dfloat2 __device__ float2
#define dfloat3 __device__ float3
#define dfloat4 __device__ float4

#define cvoid   __constant__ void
#define cint    __constant__ int
#define cuint   __constant__ unsigned int
#define cbool   __constant__ bool
#define cfloat  __constant__ float
#define cfloat2 __constant__ float2
#define cfloat3 __constant__ float3
#define cfloat4 __constant__ float4


#define FORCEINLINE __forceinline__

#define MAX_RESOURCE_COUNT 64

// TODO: structure with intrinsic vertex parameters
typedef void (*vertexShader)(unsigned int vertexID);
typedef float4 (*pixelShader)(unsigned int triIndex, void *vsOutput);

extern "C" {
	// Needed for CudaManager::testSystem();
	cint arrSize;
	gvoid adder(int *arrA, int *arrB, int *result) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, arrSize - 1);
		result[idx] = arrA[idx] + arrB[idx];
	}

	// CUDA helpers
	FORCEINLINE dfloat4 getFloat4(float x, float y, float z, float w) {
		float4 result;
		result.x = x;
		result.y = y;
		result.z = z;
		result.w = w;

		return result;
	}

	/// Pointers to UAV resources
	__constant__ float *depthBuffer;
	__constant__ float *renderTarget;
	__device__ Vertex *vertexBuffer;
	cuint *indexBuffer;
	cvoid *resources[MAX_RESOURCE_COUNT];
	__device__ vertexShader vsShader;
	__constant__ pixelShader psShader;
	cbool useDepthBuffer;

	dfloat4 fMulf4(float x, float4 v) {
		float4 result;
		result.x = v.x * x;
		result.y = v.y * x;
		result.z = v.z * x;
		result.w = v.w * x;
		return result;
	}

	dvoid vsMain(unsigned int vertexID) {
		vertexBuffer[vertexID].position = fMulf4(2.f, vertexBuffer[vertexID].position);
	}

	gvoid shadeTriangle(unsigned int index, float4 bbox, const Vertex *pts0, const Vertex *pts1, const Vertex *pts2) {
		return;
	}

	dfloat2 computeBoundingBox(float4 pos0, float4 pos1, float4 pos2) {
		float2 result;
		result.x = 0.f;
		result.y = 0.f;
		return result;
	}

	dvoid drawIndexed(const int numPrimitives) {
		// 1. RUN VS SHADER
		// Vertex = vertexShader(vertexBuffer[indexBuffer[vertexID]];
		const unsigned int primitiveID = min(blockIdx.x * blockDim.x + threadIdx.x, numPrimitives);
		const unsigned int stride = blockIdx.x * blockDim.x;

		vsShader = vsMain;
		
		for (int i = primitiveID; i < numPrimitives; i += stride) {
			vsShader(indexBuffer[i * 3 + 0]);
			vsShader(indexBuffer[i * 3 + 1]);
			vsShader(indexBuffer[i * 3 + 2]);

			// 2. FOR EACH TRIANGLE RUN WITH DYNAMIC PARALLELISM 
			// foreach (triangle) // i.e if vertexID % 3 == 0
			//   computeBoundingBox();
			//   numThreads = bbox.width * bbox.height
			//   shadeTriangleKernel<<<~numThreads>>>(triangleIndex, vsOutput)
			// end foreach
			const Vertex *pts0 = &vertexBuffer[indexBuffer[i * 3 + 0]];
			const Vertex *pts1 = &vertexBuffer[indexBuffer[i * 3 + 1]];
			const Vertex *pts2 = &vertexBuffer[indexBuffer[i * 3 + 2]];
			const float4 bbox = computeBoundingBox(pts0->position, pts1->position, pts2->position);
			const unsigned int blockSize = 192;
			const unsigned int numThreads = bbox.x * bbox.y;
			const unsigned int numBlocks = numThreads / blockSize + (numThreads % 192 != 0);
			// We want for all of the triangles bounding boxes to be completed
			__syncthreads();

			if (threadIdx.x == 0) {
				// create stream
				cudaStream_t stream;
				cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
				shadeTriangle<<<numBlocks, numThreads, 0, stream>>>(i, bbox, pts0, pts1, pts2);
				cudaDeviceSynchronize();
				cudaStreamDestroy(stream);
			}

			__syncthreads();
		}
	}

	gvoid blank(float *target, float *color, int width, int height) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, width * height - 1) * 4;

		target[idx + 0] = color[0];
		target[idx + 1] = color[1];
		target[idx + 2] = color[2];
		target[idx + 3] = color[3];
	}
}
