#include "common.hlsli"
#include "fxaa.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

float4 samplePoint(Texture2D<float4> tex, int3 p) {
	return tex.Load(p);
}

float4 blur(Texture2D<float4> tex, int3 p) {
	int3 offsets[9] = {
		int3(-1, 1, 0), int3(0, 1, 0), int3(1, 1, 0),
		int3(-1, 0, 0), int3(0, 0, 0), int3(1, 0, 0),
		int3(-1, -1, 0), int3(0, -1, 0), int3(1, -1, 0)
	};
	float kernel[9] = {
		1.f, 2.f, 1.f,
		2.f, 4.f, 2.f,
		1.f, 2.f, 1.f
	};
	float weight = 0.0625f; // 1/16

	float4 res = 0.f;
	[unroll]
	for (int i = 0; i < 9; ++i) {
		res += tex.Load(p + offsets[i]) * kernel[i];
	}

	return res * weight;
}

float4 main(PSInput IN) : SV_TARGET{
	Texture2D<float4> renderTex = ResourceDescriptorHeap[0];
	// Texture2D<float> depthTex = ResourceDescriptorHeap[1]; // TODO: use for DOF
	Texture2D<float4> hudTex = ResourceDescriptorHeap[2];
	
	int3 p = int3(IN.uv.x * sceneData.width, IN.uv.y * sceneData.height, 0);
	float4 hud = hudTex.Load(p);

	float4 color = 0.f;
	if (sceneData.fxaaON && !sceneData.darken) {
		color = fxaaFilter(renderTex, IN.uv);
		return lerp(color, hud, hud.a);
	}

	if (sceneData.darken) {
		color = blur(renderTex, p) * 0.4;
	} else {
		color = samplePoint(renderTex, p);
	}

	// tone-mapping
	color.xyz = color.xyz / (color.xyz + 1.f);

	// gamma-correction
	color.xyz = pow(color.xyz, 1/2.2f);

	// Render HUD on top
	color = lerp(color, hud, hud.a);

	return color;
}