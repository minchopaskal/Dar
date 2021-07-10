#ifndef CUDA_CPU_COMMON_H
#define CUDA_CPU_COMMON_H

#ifndef __CUDACC__
#include "d3d12_math.h"

using float4 = Vec4;
using float3 = Vec3;
using float2 = Vec2;
#endif //__CUDA_CC__

typedef struct {
	float4 position;
	/*float3 normal;
	float2 uv;*/
} Vertex;

typedef struct {
	union {
		Vertex vertices[3];
		Vertex v[3];
	};
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