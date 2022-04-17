// Includes that fix syntax highlighting
#ifdef DAR_DEBUG
#include "device_launch_parameters.h"
#include "stdio.h"
#include "math_functions.h"
#endif

#include "math_constants.h"
#include "cuda_cpu_common.h"
#include "rasterizer_utils.cuh"

typedef Vertex(*VSShader)(const Vertex*, UniformParams params);
typedef float4(*PSShader)(const Vertex*, UniformParams params);

extern "C" {
	__device__ VSShader vsShader;
	__device__ PSShader psShader;

	/// Pointers to UAV resources
	cfloat *depthBuffer;
	cfloat *renderTarget;
	cint *pixelsBarriers;
	__device__ Vertex *vertexBuffer;
	cuint *indexBuffer;
	cvoid *resources[MAX_RESOURCES_COUNT];
	cfloat clearColor[4];
	cbool useDepthBuffer;
	__constant__ CudaRasterizerCullType cullType;
	cuint width;
	cuint height;

	FORCEINLINE dfloat getDiscreetValue(const float value, const unsigned int steps) {
		return (int)min(max((value + 1.f) * 0.5f * steps, 0.f), float(steps));
	}

	FORCEINLINE dfloat2 getDiscreeteCoordinates(const float4 p, const unsigned int width, const unsigned int height) {
		float2 result;
		// [-1.f, 1.f] -> [0; width] OR [0; height]
		result.x = getDiscreetValue(p.x, width);
		result.y = getDiscreetValue(p.y, height);

		return result;
	}

	FORCEINLINE dfloat3 findBarys(
		const float2 p0,
		const float2 p1,
		const float2 p2,
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

	/// Computes bounding box of a triangle given its coordinates.
	/// Bounding box layout:
	/// float4(bbox.topLeftXCoordinate, bbox.topLeftYCoordinate, bbox.height, bbox.width)
	dfloat4 computeBoundingBox(
		const float2 pos0,
		const float2 pos1,
		const float2 pos2
	) {
		float4 result;
		result.x = min(min(pos0.x, pos1.x), pos2.x);
		result.y = min(min(pos0.y, pos1.y), pos2.y);
		result.z = (int)max(max(pos0.y, pos1.y), pos2.y) - result.y;
		result.w = (int)max(max(pos0.x, pos1.x), pos2.x) - result.x;

		return result;
	}

	FORCEINLINE dfloat4 baryInterpolationf4(const float3 barys, const float4 v1, const float4 v2, const float4 v3) {
		return make_float4(
			fmaf(barys.x, v1.x, fmaf(barys.y, v2.x, fmaf(barys.z, v3.x, 0.f))),
			fmaf(barys.x, v1.y, fmaf(barys.y, v2.y, fmaf(barys.z, v3.y, 0.f))),
			fmaf(barys.x, v1.z, fmaf(barys.y, v2.z, fmaf(barys.z, v3.z, 0.f))),
			fmaf(barys.x, v1.w, fmaf(barys.y, v2.w, fmaf(barys.z, v3.w, 0.f)))
		);
	}

	FORCEINLINE dfloat3 baryInterpolationf3(const float3 barys, const float3 v1, const float3 v2, const float3 v3) {
		return make_float3(
			fmaf(barys.x, v1.x, fmaf(barys.y, v2.x, fmaf(barys.z, v3.x, 0.f))),
			fmaf(barys.x, v1.y, fmaf(barys.y, v2.y, fmaf(barys.z, v3.y, 0.f))),
			fmaf(barys.x, v1.z, fmaf(barys.y, v2.z, fmaf(barys.z, v3.z, 0.f)))
		);
	}

	FORCEINLINE dfloat2 baryInterpolationf2(const float3 barys, const float2 v1, const float2 v2, const float2 v3) {
		return make_float2(
			fmaf(barys.x, v1.x, fmaf(barys.y, v2.x, fmaf(barys.z, v3.x, 0.f))),
			fmaf(barys.x, v1.y, fmaf(barys.y, v2.y, fmaf(barys.z, v3.y, 0.f)))
		);
	}

	FORCEINLINE __device__ Vertex getInterpolatedVertex(float3 barys, Vertex v0, Vertex v1, Vertex v2) {
		Vertex result;
		result.position = baryInterpolationf4(barys, v0.position, v1.position, v2.position);
		result.normal = baryInterpolationf3(barys, v0.normal, v1.normal, v2.normal);
		result.uv = baryInterpolationf2(barys, v0.uv, v1.uv, v2.uv);

		return result;
	}

	// Rasterization functions
	gvoid shadeTriangle(
		UniformParams params,
		const unsigned int primitiveID,
		const unsigned int numPrimitives,
		const float4 bbox,
		const Vertex v0,
		const Vertex v1,
		const Vertex v2,
		const float2 dp0, // screen-space coordinates of the vertices. to-do: change name
		const float2 dp1, // screen-space coordinates of the vertices. to-do: change name
		const float2 dp2, // screen-space coordinates of the vertices. to-do: change name
		const float3 edge0,
		const float3 edge1,
		const float3 edge2,
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

			const float dX = p.x - bbox.x;
			const float dY = p.y - bbox.y;
			const float edge0Eq = fmaf(dX, edge0.y, fmaf(-dY, edge0.x, edge0.z));
			const float edge1Eq = fmaf(dX, edge1.y, fmaf(-dY, edge1.x, edge1.z));
			const float edge2Eq = fmaf(dX, edge2.y, fmaf(-dY, edge2.x, edge2.z));

			// Check if point is inside the triangle by checking it agains the
			// edge equations of the triangle edges.
			if (edge0Eq > 0 || edge1Eq > 0 || edge2Eq > 0) {
				continue;
			}

			const unsigned int y = height - p.y - 1;
			const unsigned int pixelIndex = (y * width + p.x);
			const unsigned int pixelIndexOffset = pixelIndex * 4;

			//assert(pixelIndex < width * height);

			// TODO: do not do the culling here
			if (pixelIndex >= width * height * 4) {
				return;
			}

			// Transform vertex NDC coordinates to screen coords and
			// find barys of shaded pixel based on that
			const float3 barys = findBarys(dp0, dp1, dp2, p);
			Vertex interpolatedVertex = getInterpolatedVertex(barys, v0, v1, v2);

			bool passDepthTest = true;
			while (true) {
				if (!useDepthBuffer) {
					break;
				}

				passDepthTest = false;
				float oldZ = depthBuffer[pixelIndex];
				if (oldZ < interpolatedVertex.position.z) {
					break;
				}

				passDepthTest = true;
				if (atomicCAS(
						(unsigned int*)&depthBuffer[pixelIndex],
						__float_as_uint(oldZ),
						__float_as_uint(interpolatedVertex.position.z)) == __float_as_uint(oldZ)) {
					break;
				}
			}

			if (!passDepthTest) {
				continue;
			}

			float4 color = psShader(&interpolatedVertex, params);
			renderTarget[pixelIndexOffset + 0] = color.x;
			renderTarget[pixelIndexOffset + 1] = color.y;
			renderTarget[pixelIndexOffset + 2] = color.z;
			renderTarget[pixelIndexOffset + 3] = color.w;
		}
	}

	gvoid processVertices(const unsigned int numVertices, const unsigned int width, const unsigned int height) {
		const unsigned int vertexID = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int stride = gridDim.x * blockDim.x;

		if (vertexID >= numVertices) {
			return;
		}

		UniformParams params;
		params.resources = resources;
		params.width = width;
		params.height = height;

		for (int i = vertexID; i < numVertices; i += stride) {
			Vertex res = vsShader(&vertexBuffer[i], params); // vertex shading
			res.position /= res.position.w; // perspective division
			vertexBuffer[i] = res;
		}
	}

	gvoid drawIndexed(const int numPrimitives, const unsigned int width, const unsigned int height) {
		const unsigned int primitiveID = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int stride = gridDim.x * blockDim.x;

		if (primitiveID >= numPrimitives) {
			return;
		}

		UniformParams params;
		params.resources = resources;
		params.width = width;
		params.height = height;

		for (int i = primitiveID; i < numPrimitives; i += stride) {
			Vertex v0 = vertexBuffer[indexBuffer[i * 3 + 0]];
			Vertex v1 = vertexBuffer[indexBuffer[i * 3 + 1]];
			Vertex v2 = vertexBuffer[indexBuffer[i * 3 + 2]];

			// Back-face culling
			// Vertices are now in NDC, so we can test for back-face against
			// the (0, 0, -1) vector which points outside the monitor.
			// TODO: if (cull)
			const float3 ab = fromFloat4(v1.position - v0.position);
			const float3 ac = fromFloat4(v2.position - v0.position);
			const float3 normal = normalize(cross(ac, ab));
			if (cullType != cullType_none) {
				const bool backface = dot(normal, make_float3(0.f, 0.f, -1.f)) < 0;
				if ((cullType == cullType_backface && backface) || (cullType == cullType_frontface && !backface)) {
					continue;
				}
			}

			// TODO: write rasterization in hierarchical approach
			// for each triangle run 1 thread for each 8x8(or smth else) block
			// inside its bounding box. The thread should quickly test if the block 
			// is inside the triangle. Mark the blocks that contain part of the triangle.
			// Second level would be to run threads for each marked block and rasterize
			// the part of the triangle inside.
			
			// 2. FOR EACH TRIANGLE RUN WITH DYNAMIC PARALLELISM 
			// foreach (triangle) // i.e if vertexID % 3 == 0
			//   computeBoundingBox();
			//   numThreads = bbox.width * bbox.height
			//   shadeTriangleKernel<<<~numThreads>>>(triangleIndex, vsOutput)
			// end foreach
			const float2 dp0 = getDiscreeteCoordinates(v0.position, width, height);
			const float2 dp1 = getDiscreeteCoordinates(v1.position, width, height);
			const float2 dp2 = getDiscreeteCoordinates(v2.position, width, height);

			const float2 e0 = dp1 - dp0;
			const float2 e1 = dp2 - dp1;
			const float2 e2 = dp0 - dp2;

			const float4 bbox = computeBoundingBox(dp0, dp1, dp2);
			// Layout of edge vectors:
			// (dx, dy, edge equation for top-left bbox corner)
			const float3 edge0 = make_float3(e0.x, e0.y, (bbox.x - dp0.x) * e0.y - (bbox.y - dp0.y) * e0.x);
			const float3 edge1 = make_float3(e1.x, e1.y, (bbox.x - dp1.x) * e1.y - (bbox.y - dp1.y) * e1.x);
			const float3 edge2 = make_float3(e2.x, e2.y, (bbox.x - dp2.x) * e2.y - (bbox.y - dp2.y) * e2.x);

			const unsigned int blockSize = 128;
			const unsigned int numThreads = bbox.z * bbox.w;
			const unsigned int numBlocks = (numThreads / blockSize) + (numThreads % blockSize != 0);

			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			shadeTriangle<<<numBlocks, blockSize, 0, stream>>>(params, i, numPrimitives, bbox, v0, v1, v2, dp0, dp1, dp2, edge0, edge1, edge2, width, height);
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
