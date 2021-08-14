#ifndef CUDA_CPU_COMMON_H
#define CUDA_CPU_COMMON_H

#ifndef __CUDACC__
#include "d3d12_math.h"

using float2 = Vec2;
using float3 = Vec3;
using float4 = Vec4;

using int2 = Vec2i;
using int3 = Vec3i;
using int4 = Vec4i;

#endif //__CUDA_CC__

#define MAX_RESOURCES_COUNT 64

struct Vertex {
	float4 position;
	/*float3 normal;
	float2 uv;*/
};

struct Triangle {
	Vertex vertices[3];
};

enum PrimitiveType {
	primitiveType_triangle,

	primitiveType_count
};

enum CudaRasterizerCullType {
	cullType_none,
	cullType_backface,
	cullType_frontface,

	cullType_count
};

struct CUDAShaderPointers {
	void *vsShaderPtr;
	void *psShaderPtr;
};

#endif // CUDA_CPU_COMMON_H