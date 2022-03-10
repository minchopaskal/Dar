#include "common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

float4 main(PSInput IN) : SV_TARGET {
	Texture2D<float4> tex = ResourceDescriptorHeap[0];

	return tex.Sample(Sampler, IN.uv);
}