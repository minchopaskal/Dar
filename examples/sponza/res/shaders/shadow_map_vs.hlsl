#include "common.hlsli"

struct VSInput
{
	float3 position : POSITION0;
	float3 normal : NORMAL;
	float3 tangent : TANGENT;
	float2 uv : TEXCOORD;
};

struct VSOutput
{
	float4 position : SV_Position;
	float2 uv : TEXCOORD;
};

ConstantBuffer<LightcasterDesc> lightcaster: register(b2);

static const uint LIGHTS_BUFFER_INDEX = 0;

VSOutput main(VSInput IN)
{
	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[LIGHTS_BUFFER_INDEX];

	VSOutput result;
	
	float4 fragPos = mul(meshData.modelMatrix, float4(IN.position, 1.f));
	result.position = mul(lights[lightcaster.index].viewProjection, fragPos);
	result.uv = IN.uv;

	return result;
}
