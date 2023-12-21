#include "hud_common.hlsli"

struct PSInput {
	float4 pos : SV_Position;
	float2 uv: TEXCOORD;
	nointerpolation uint index : BLENDINDICES0; // why the hell not
};

SamplerState Sampler : register(s0);

static const uint WIDGET_DATA_INDEX = 0;
static const uint TEXTURES_OFFSET = 1;

float roundedBox(float2 uv, float radius) {
	return length(max(abs(uv)-1.f+radius,0.0))-radius;
}

float4 main(PSInput IN) : SV_Target {
	StructuredBuffer<WidgetData> widgetData = ResourceDescriptorHeap[WIDGET_DATA_INDEX]; 
	WidgetData data = widgetData[IN.index];

	if (data.textureIndex >= 0) {
		Texture2D<float4> tex = ResourceDescriptorHeap[data.textureIndex + TEXTURES_OFFSET];
		data.color = tex.Sample(Sampler, IN.uv);
	}

	float2 uv = float2(IN.uv.x * 2. - 1., IN.uv.y * -2.f + 1.f);
	float box = -roundedBox(uv, .3f);
	float sbox = smoothstep(0., 0.05f, box); // anti-aliasing
	return sbox * data.color;
}
