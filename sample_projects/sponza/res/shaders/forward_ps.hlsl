#include "lighting_common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
};

static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint MATERIALS_BUFFER_INDEX = 1;
static const uint TEXTURE_BUFFERS_START = 2;

float4 main(PSInput IN) : SV_TARGET{
	// TODO
	return float4(1.f, 0.f, 0.f, 1.f);
}