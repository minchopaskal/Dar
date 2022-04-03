#include "common.hlsli"
#include "fxaa.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

float4 main(PSInput IN) : SV_TARGET{
	Texture2D<float4> renderTex = ResourceDescriptorHeap[0];
	
	// TODO: use for DoF
	// Texture2D<float> depthTex = ResourceDescriptorHeap[1];

	if (sceneData.fxaaON) {
		return fxaaFilter(renderTex, IN.uv);
	}

	int3 p = int3(IN.uv.x * sceneData.width, IN.uv.y * sceneData.height, 0);
	float4 color = renderTex.Load(p);

	// tone-mapping
	color = color / (color + 1.f);

	// apply gamma-correction
	color = pow(color, 1 / 2.2);

	return color;
}