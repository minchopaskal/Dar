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
#define gint2   __global__ int2
#define gint3   __global__ int3
#define gint4   __global__ int4
#define guint   __global__ unsigned int
#define gbool   __global__ bool
#define gfloat  __global__ float
#define gfloat2 __global__ float2
#define gfloat3 __global__ float3
#define gfloat4 __global__ float4

#define dvoid   __device__ void
#define dint    __device__ int
#define dint2   __device__ int2
#define dint3   __device__ int3
#define dint4   __device__ int4
#define duint   __device__ unsigned int
#define dbool   __device__ bool
#define dfloat  __device__ float
#define dfloat2 __device__ float2
#define dfloat3 __device__ float3
#define dfloat4 __device__ float4

#define cvoid   __constant__ void
#define cint    __constant__ int
#define cint2   __constant__ int2
#define cint3   __constant__ int3
#define cint4   __constant__ int4
#define cuint   __constant__ unsigned int
#define cbool   __constant__ bool
#define cfloat  __constant__ float
#define cfloat2 __constant__ float2
#define cfloat3 __constant__ float3
#define cfloat4 __constant__ float4

#define FORCEINLINE __forceinline__

// TODO: create shder input/output layout:
/*
InOutLayout{
	float4 elements[MAX_ELEMENTS_COUNT];
	int numElements;
}
*/
// Should be one per vertex. Pixel shader should receive that 

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

	// TODO: Separate them in another file. This means CUDABase should support compilation of
	// multiple files.
	// CUDA helpers / Math functions
	FORCEINLINE dfloat3 getFloat3(float x, float y, float z) {
		float3 result;
		result.x = x;
		result.y = y;
		result.z = z;

		return result;
	}

	FORCEINLINE dfloat4 getFloat4(float x, float y, float z, float w) {
		float4 result;
		result.x = x;
		result.y = y;
		result.z = z;
		result.w = w;

		return result;
	}

	FORCEINLINE dfloat3 cross(const float3 v1, const float3 v2) {
		return getFloat3(
			v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x
		);
	}

	FORCEINLINE dfloat4 fMulf4(float x, float4 v) {
		float4 result;
		result.x = v.x * x;
		result.y = v.y * x;
		result.z = v.z * x;
		result.w = v.w * x;
		return result;
	}

	FORCEINLINE dfloat4 f4Plusf4(float4 v1, float4 v2) {
		float4 result;
		result.x = v1.x * v2.x;
		result.y = v1.y * v2.y;
		result.z = v1.z * v2.z;
		result.w = v1.w * v2.w;

		return result;
	}

	FORCEINLINE dint2 i2Minusi2(int2 v1, int2 v2) {
		int2 result;
		result.x = v1.x - v2.x;
		result.y = v1.y - v2.y;

		return result;
	}

	/// Pointers to UAV resources
	cfloat *depthBuffer;
	cfloat *renderTarget;
	cint *pixelsBarriers;
	__device__ Vertex *vertexBuffer;
	cuint *indexBuffer;
	cvoid *resources[MAX_RESOURCES_COUNT];
	cbool useDepthBuffer;

	// These should be constants
	dvoid vsMain(unsigned int vertexID) {
		//vertexBuffer[vertexID].position = fMulf4(2.f, vertexBuffer[vertexID].position);
	}

	dfloat4 psMain(unsigned int vertexID, void* /*vsOutput*/) {
		return getFloat4(1.f, 0.f, 0.f, 1.f);
	}

	__device__ vertexShader vsShader = vsMain;
	__device__ pixelShader psShader = psMain;

	FORCEINLINE dint getDiscreetValue(float value, const unsigned int steps) {
		return (int)min(max((value + 1.f) * 0.5f * steps, 0.f), float(steps));
	}

	FORCEINLINE dint2 getDiscreeteCoordinates(float4 p, const unsigned int width, const unsigned int height) {
		int2 result;
		// [-1.f, 1.f] -> [0; width/height]
		result.x = getDiscreetValue(p.x, width);
		result.y = getDiscreetValue(p.y, height);

		return result;
	}

	FORCEINLINE dfloat3 findBarys(
		const int2 p0,
		const int2 p1,
		const int2 p2,
		const int2 p
	) {
		// TODO: Crammer's rule but compacted. Test perf with classic
		float3 u = cross(
			getFloat3(p1.x - p0.x, p2.x - p0.x, p0.x - p.x),
			getFloat3(p1.y - p0.y, p2.y - p0.y, p0.y - p.y)
		);

		if (fabs(u.z) < 1.f) {
			return getFloat3(-1.f, 1.f, 1.f);
		}

		float3 res;
		res.y = u.x / u.z;
		res.z = u.y / u.z;
		res.x = 1.f - (res.y + res.z);
		return res;
	}

	FORCEINLINE __device__ Vertex getInterpolatedVertex(
		const float3 barys,
		const Vertex *pts0,
		const Vertex *pts1,
		const Vertex *pts2
	) {
		Vertex result;
		result.position = f4Plusf4(fMulf4(barys.x, pts0->position), f4Plusf4(fMulf4(barys.y, pts1->position), fMulf4(barys.z, pts2->position)));
		return result;
	}

	/// Computes bounding box of a triangle given its coordinates.
	/// Bounding box layout:
	/// float4(bbox.topLeftXCoordinate, bbox.topLeftYCoordinate, bbox.height, bbox.width)
	dint4 computeBoundingBox(
		float4 pos0,
		float4 pos1,
		float4 pos2,
		const unsigned int width,
		const unsigned int height
	) {
		int4 result;
		result.x = getDiscreetValue(min(min(pos0.x, pos1.x), pos2.x), width);
		result.y = getDiscreetValue(min(min(pos0.y, pos1.y), pos2.y), height);
		result.z = getDiscreetValue(max(max(pos0.y, pos1.y), pos2.y), height) - result.y;
		result.w = getDiscreetValue(max(max(pos0.x, pos1.x), pos2.x), width) - result.x;

		return result;
	}

	// Rasterization functions
	gvoid shadeTriangle(
		unsigned int primitiveID,
		unsigned int numPrimitives,
		const int4 bbox,
		const Vertex *pts0,
		const Vertex *pts1,
		const Vertex *pts2,
		const unsigned int width,
		const unsigned int height
	) {
		const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int stride = gridDim.x * blockDim.x;
		const unsigned int pixelsInBBox = bbox.z * bbox.w;

		const unsigned int bboxWidth = bbox.w;

		//printf("Launched child kernel! blockid: %d, blockdim: %d, threadid: %d TheadID: %d\n", blockIdx.x, blockDim.x, threadIdx.x, threadID);

		for (int i = threadID; i < pixelsInBBox; i += stride) {
			// Vertex positions are now in NDC so find barys based on NDC
			// Only after that, will we compute the discreete device coordinates.
			int2 p;
			p.x = threadID % bboxWidth + bbox.x;
			p.y = threadID / bboxWidth + bbox.y;

			const unsigned int pixelIndex = (p.y * width + p.x) * 4;

			// TODO: do not do the culling here
			if (pixelIndex >= width * height * 4) {
				return;
			}

			float3 barys = findBarys(
				getDiscreeteCoordinates(pts0->position, width, height),
				getDiscreeteCoordinates(pts1->position, width, height),
				getDiscreeteCoordinates(pts2->position, width, height),
				p
			);

			if (
				barys.x < 0.f || barys.x > 1.f ||
				barys.y < 0.f || barys.y > 1.f ||
				barys.z < 0.f || barys.z > 1.f ||
				fabs(barys.x + barys.y + barys.z - 1.f) > 1e-6f) {
				continue;
			}

			Vertex interpolated = getInterpolatedVertex(barys, pts0, pts1, pts2);
			
			float4 color = psShader(primitiveID, &interpolated);
			// color renderTarget[x, y] with color returned from psShader

			////while (atomicExch(&pixelsBarriers[pixelIndex], 1)) { /* busyWait */ }

			// TODO: depth test
			renderTarget[pixelIndex + 0] = color.x;
			renderTarget[pixelIndex + 1] = color.y;
			renderTarget[pixelIndex + 2] = color.z;
			renderTarget[pixelIndex + 3] = color.w;
			//atomicExch(&pixelsBarriers[pixelIndex], 0);
		}
	}

	gvoid drawIndexed(const int numPrimitives, int width, int height) {
		const unsigned int primitiveID = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int stride = gridDim.x * blockDim.x;

		if (primitiveID >= numPrimitives) {
			return;
		}

		for (int i = primitiveID; i < numPrimitives; i += stride) {
			// 1. RUN VS SHADER
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
			const int4 bbox = computeBoundingBox(pts0->position, pts1->position, pts2->position, width, height);
			const unsigned int blockSize = 192;
			const unsigned int numThreads = bbox.z * bbox.w;
			const unsigned int numBlocks = (numThreads / blockSize) + (numThreads % 192 != 0);

			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			shadeTriangle<<<numBlocks, blockSize, 0, stream>>>(i, numPrimitives, bbox, pts0, pts1, pts2, width, height);
			cudaDeviceSynchronize();
			cudaStreamDestroy(stream);
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
