// Includes that fix syntax highlighting
#ifdef D3D12_DEBUG
#include "device_launch_parameters.h"
#include "stdio.h"
#include "math_functions.h"
#endif

#include "math_constants.h"

#include "cuda_cpu_common.h"
#include "rasterizer_utils.cu"

typedef Vertex (*vertexShader)(unsigned int vertexID);
typedef float4 (*pixelShader)(unsigned int triIndex, void *vsOutput, float3 barys);

extern "C" {
	/// Pointers to UAV resources
	cfloat *depthBuffer;
	cfloat *renderTarget;
	cint *pixelsBarriers;
	__device__ Vertex *vertexBuffer;
	cuint *indexBuffer;
	cvoid *resources[MAX_RESOURCES_COUNT];
	cfloat clearColor[4];
	cbool useDepthBuffer;
	__device__ CudaRasterizerCullType cullType;
	cuint width;
	cuint height;

	// These should be constants
	__device__ Vertex vsMain(unsigned int vertexID) {
		Vertex *v = &vertexBuffer[vertexID];
		Vertex result;
		result.position = v->position;
		result.position.z = -result.position.z;
		return result;
	}

	dfloat4 psMain(unsigned int primID, void *vsOutput, float3 barys) {
		Vertex *v = (Vertex*)vsOutput;

		// Z is in NDC, i.e [-1.f; 1.f]. We need a value between [0.f; 1.f], thus the transformation.
		// Since Z increases as depth increases we subtract the value from 1
		// so that nearer objects appear brighter.
		float shade = 1.f - (v->position.z + 1.f) * 0.5f;
		return make_float4(shade, shade, shade, 1.f);
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
			make_float3(p1.x - p0.x, p2.x - p0.x, p0.x - p.x),
			make_float3(p1.y - p0.y, p2.y - p0.y, p0.y - p.y)
		);

		if (fabs(u.z) < 1.f) {
			return make_float3(-1.f, 1.f, 1.f);
		}

		float3 res;
		res.y = u.x / u.z;
		res.z = u.y / u.z;
		res.x = 1.f - (res.y + res.z);
		return res;
	}

	FORCEINLINE __device__ Vertex getInterpolatedVertex(
		const float3 barys,
		const Vertex pts0,
		const Vertex pts1,
		const Vertex pts2
	) {
		Vertex result;
		result.position = barys.x * pts0.position + barys.y * pts1.position + barys.z * pts2.position;

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
		const Vertex pts0,
		const Vertex pts1,
		const Vertex pts2,
		float *depthBuffer,
		const unsigned int width,
		const unsigned int height
	) {
		const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int stride = gridDim.x * blockDim.x;
		const unsigned int pixelsInBBox = bbox.z * bbox.w;

		const unsigned int bboxWidth = bbox.w;

		for (int i = threadID; i < pixelsInBBox; i += stride) {
			int2 p;
			p.x = threadID % bboxWidth + bbox.x;
			p.y = threadID / bboxWidth + bbox.y;

			const unsigned int y = height - p.y - 1;
			const unsigned int pixelIndex = (y * width + p.x);
			const unsigned int pixelIndexOffset = pixelIndex * 4;

			assert(pixelIndex < width * height);

			// TODO: do not do the culling here
			if (pixelIndex >= width * height * 4) {
				return;
			}

			// Transform vertex NDC coordinates to screen coords and
			// find barys of shaded pixel based on that
			float3 barys = findBarys(
				getDiscreeteCoordinates(pts0.position, width, height),
				getDiscreeteCoordinates(pts1.position, width, height),
				getDiscreeteCoordinates(pts2.position, width, height),
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

			bool passDepthTest = true;
			while (true) {
				if (!useDepthBuffer) {
					break;
				}

				passDepthTest = false;
				float oldZ = depthBuffer[pixelIndex];
				if (oldZ < interpolated.position.z) {
					break;
				}

				passDepthTest = true;
				if (atomicCAS(
						(unsigned int*)&depthBuffer[pixelIndex],
						__float_as_uint(oldZ),
						__float_as_uint(interpolated.position.z)) == __float_as_uint(oldZ)) {
					break;
				}
			}

			if (!passDepthTest) {
				continue;
			}

			float4 color = psShader(primitiveID, &interpolated, barys);
			renderTarget[pixelIndexOffset + 0] = color.x;
			renderTarget[pixelIndexOffset + 1] = color.y;
			renderTarget[pixelIndexOffset + 2] = color.z;
			renderTarget[pixelIndexOffset + 3] = color.w;
		}
	}

	gvoid drawIndexed(const int numPrimitives, int width, int height) {
		const unsigned int primitiveID = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int stride = gridDim.x * blockDim.x;

		if (primitiveID >= numPrimitives) {
			return;
		}

		for (int i = primitiveID; i < numPrimitives; i += stride) {
			// 1. RUN VS shader
			const Vertex pts0 = vsShader(indexBuffer[i * 3 + 0]);
			const Vertex pts1 = vsShader(indexBuffer[i * 3 + 1]);
			const Vertex pts2 = vsShader(indexBuffer[i * 3 + 2]);

			// Back-face culling
			// Vertices are now in NDC, so we can test for back-face against
			// the (0, 0, -1) vector which points outside the monitor.
			// TODO: if (cull)
			if (cullType != cullType_none) {
				float3 ab = fromFloat4(pts1.position - pts0.position);
				float3 ac = fromFloat4(pts2.position - pts0.position);
				float3 normal = normalize(cross(ac, ab));
				bool backface = dot(normal, make_float3(0.f, 0.f, -1.f)) < 0;
				if ((cullType == cullType_backface && backface) || (cullType == cullType_frontface && !backface)) {
					continue;
				}
			}

			// 2. FOR EACH TRIANGLE RUN WITH DYNAMIC PARALLELISM 
			// foreach (triangle) // i.e if vertexID % 3 == 0
			//   computeBoundingBox();
			//   numThreads = bbox.width * bbox.height
			//   shadeTriangleKernel<<<~numThreads>>>(triangleIndex, vsOutput)
			// end foreach
			const int4 bbox = computeBoundingBox(pts0.position, pts1.position, pts2.position, width, height);
			const unsigned int blockSize = 256;
			const unsigned int numThreads = bbox.z * bbox.w;
			const unsigned int numBlocks = (numThreads / blockSize) + (numThreads % blockSize != 0);

			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			shadeTriangle<<<numBlocks, blockSize, 0, stream>>>(i, numPrimitives, bbox, pts0, pts1, pts2, depthBuffer, width, height);
			cudaStreamDestroy(stream);
		}
	}

	gvoid blank(float *target, int width, int height) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, width * height - 1) * 4;

		target[idx + 0] = clearColor[0];
		target[idx + 1] = clearColor[1];
		target[idx + 2] = clearColor[2];
		target[idx + 3] = clearColor[3];
	}
}
