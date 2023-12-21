#include "hud_common.hlsli"

struct VSInput {
	float2 pos : POSITION0;
	float2 uv: TEXCOORD;
};

struct VSOut {
	float4 pos : SV_Position;
	float2 uv: TEXCOORD;
	nointerpolation uint index : BLENDINDICES0; // why the hell not
};

ConstantBuffer<HUDConstData> constData : register(b0);

VSOut main(VSInput IN, in uint vertID: SV_VertexID) {
	VSOut result;
	result.pos =  mul(constData.projection, float4(IN.pos, 0.f, 1.f));
	result.uv = IN.uv;
	result.index = vertID >> 2;

	return result;
}
