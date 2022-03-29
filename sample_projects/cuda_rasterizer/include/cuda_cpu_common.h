#ifndef CUDA_CPU_COMMON_H
#define CUDA_CPU_COMMON_H

#ifndef __CUDACC__
#include "math/dar_math.h"

using float2 = Vec2;
using float3 = Vec3;
using float4 = Vec4;

using int2 = Vec2i;
using int3 = Vec3i;
using int4 = Vec4i;

#endif //__CUDA_CC__

#ifdef _WIN64
#define ALIGNAS(x) __declspec(align((x)))
#elif defined(__CUDACC__)
#define ALIGNAS(x) __alignas__((x))
#endif

#define MAX_RESOURCES_COUNT 64

// Alignment requirements from CUDA
struct Vertex {
	ALIGNAS(16) float4 position;
	float3 normal;
	ALIGNAS(8) float2 uv;
};
static_assert(sizeof(Vertex) == 48, "Invalid size for struct Vertex. Check alignment!");

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

// Parameters passed to both shader stages
struct UniformParams {
	void *const *resources;
	unsigned int width;
	unsigned int height;
};

struct ALIGNAS(16) TextureSampler {
	unsigned char *data;
	int width;
	int height;
	int numComp;
};
static_assert(alignof(TextureSampler) == 16, "Misaligned TextureData structure!");

#endif // CUDA_CPU_COMMON_H