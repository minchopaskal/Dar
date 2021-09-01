#include "rasterizer_utils.cuh"

FORCEINLINE __device__ Vertex vsBasicShader(const Vertex *v, UniformParams params) {
	Vertex result;
	result.position = v->position;
	result.normal = v->normal;
	result.uv = v->uv;

	mat4 *modelMat = ( mat4* )(params.resources[0]);
	mat4 *viewMat = ( mat4* )(params.resources[1]);
	mat4 *perspective = ( mat4* )(params.resources[2]);
	mat4 *normalMat = ( mat4* )(params.resources[3]);

	// model transform. The model's coordinates
	// was designed for RH system, so reverse the z coord.
	result.position = *modelMat * result.position;
	result.position = *viewMat * result.position;
	result.position = *perspective * result.position;

	result.normal = fromFloat4(*normalMat * toFloat4(result.normal));

	return result;
}

FORCEINLINE __device__ float4 psBasicShader(const Vertex *v, UniformParams params) {
	// ====== Depth shading
	// Z is in NDC, i.e [-1.f; 1.f]. We need a value between [0.f; 1.f], thus the transformation.
	// Since Z increases as depth increases we subtract the value from 1
	// so that nearer objects appear brighter.
	//const float shade = 1.f - (v->position.z + 1.f) * 0.5f;

	// ====== Gourad shading
	const float shade = dot(v->normal, make_float3(0.f, 0.f, 1.f));
	//return make_float4(shade, shade, shade, 1.f);

	TextureSampler *tex = ( TextureSampler* )(params.resources[4]);
	float4 res = sample(tex, v->uv) * shade;
	res.w = 1.f;

	return res;
}

extern "C" {
	__global__ void getShaderPtrs_BasicShader(CUDAShaderPointers *shaderPtrs) {
		shaderPtrs->vsShaderPtr = &vsBasicShader;
		shaderPtrs->psShaderPtr = &psBasicShader;
	}
}
