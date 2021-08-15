#include "cuda_cpu_common.h"
#include "rasterizer_utils.cuh"

FORCEINLINE __device__ Vertex vsBasicShader(const Vertex *v) {
	Vertex result;
	result.position = v->position;
	result.normal = v->normal;
	result.uv = v->uv;
	result.position.z = -result.position.z;

	return result;
}

FORCEINLINE __device__ float4 psBasicShader(const Vertex *v) {
	// Z is in NDC, i.e [-1.f; 1.f]. We need a value between [0.f; 1.f], thus the transformation.
	// Since Z increases as depth increases we subtract the value from 1
	// so that nearer objects appear brighter.
	const float shade = 1.f - (v->position.z + 1.f) * 0.5f;

	//const float shade = dot(v->normal, make_float3(0.f, 0.f, 1.f));

	return make_float4(shade, shade, shade, 1.f);
}

extern "C" {
	__global__ void getShaderPtrs_BasicShader(CUDAShaderPointers *shaderPtrs) {
		shaderPtrs->vsShaderPtr = &vsBasicShader;
		shaderPtrs->psShaderPtr = &psBasicShader;
	}
}
