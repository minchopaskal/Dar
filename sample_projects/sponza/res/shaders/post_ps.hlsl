#include "common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

float4 main(PSInput IN) : SV_TARGET{
	Texture2D<float4> renderTex = ResourceDescriptorHeap[0];
	Texture2D<float> depthTex = ResourceDescriptorHeap[1];

	int3 p = int3(IN.uv.x * sceneData.width, IN.uv.y * sceneData.height, 0);
	float4 color = renderTex.Load(p);

	// apply gamma-correction
	return pow(color, 1/2.2);
}