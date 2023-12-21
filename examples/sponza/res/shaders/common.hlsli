#ifndef COMMON_HLSLI
#define COMMON_HLSLI

#include "gpu_cpu_common.hlsli"

#define PI 3.14159265f

SamplerState Sampler : register(s0);
SamplerComparisonState SamplerDepth : register(s1);

ConstantBuffer<ShaderRenderData> sceneData : register(b0);
ConstantBuffer<MeshData> meshData : register(b1);

static const uint INVALID_MATERIAL_INDEX = 0xffffffff;

float4 getColorFromTexture(uint textureIndex, uint indexOffset, float2 uv, TextureUsage usage = TextureUsage::Default, float4 defaultColor = float4(1.f, 1.f, 1.f, 1.f)) {
	float4 color = defaultColor;
	if (textureIndex != INVALID_MATERIAL_INDEX) {
		Texture2D<float4> tex = ResourceDescriptorHeap[textureIndex + indexOffset];
		
		color = tex.Sample(Sampler, uv);
		// if (usage == TextureUsage::NormalMap) {
		// 	color.xyz = normalize(color.xyz);
		// }
	}

	return color;
}

#endif // COMMON_HLSLI
