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

typedef struct {
	float4 position;
	/*float3 normal;
	float2 uv;*/
} Vertex;

typedef struct {
	Vertex vertices[3];
} Triangle;

typedef struct {
	float4 positions[3];
	/*float3 normals[3];
	float2 uvs[3];*/
} TriangleDOD;

typedef enum {
	primitiveType_triangle,

	primitiveType_count
} PrimitiveType;

#endif // CUDA_CPU_COMMON_H