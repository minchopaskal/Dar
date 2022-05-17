#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "gpu_cpu_common.hlsli"

#define PI 3.14159265f

SamplerState Sampler : register(s0);
ConstantBuffer<ShaderRenderData> sceneData : register(b0);
ConstantBuffer<MeshData> meshData : register(b1);

static const uint INVALID_MATERIAL_INDEX = 0xffffffff;

float4 getColorFromTexture(uint textureIndex, uint indexOffset, float2 uv, float4 defaultColor = float4(1.f, 1.f, 1.f, 1.f)) {
	float4 color = defaultColor;
	if (textureIndex != INVALID_MATERIAL_INDEX) {
		Texture2D<float4> tex = ResourceDescriptorHeap[textureIndex + indexOffset];
		color = tex.Sample(Sampler, uv);
	}

	return color;
}

#endif // COMMON_HLSLI
