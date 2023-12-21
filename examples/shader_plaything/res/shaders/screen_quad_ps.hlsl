#include "common.hlsli"

float4 main(PSInput IN) : SV_TARGET
{
	if (!constData.hasOutput) {
		return 0.f;
	}

	Texture2D<float4> renderTex = ResourceDescriptorHeap[0];
	return renderTex.Sample(Sampler, IN.uv);
}