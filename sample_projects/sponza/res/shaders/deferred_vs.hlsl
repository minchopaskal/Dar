#include "common.hlsli"

struct VSInput
{  
	float3 position : POSITION0;
	float3 normal: NORMAL;
	float2 uv : TEXCOORD;
};

struct VSOutput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
	float4 position : SV_Position;
};

VSOutput main(VSInput IN)
{
	VSOutput result;
	
	result.uv = IN.uv;
	result.normal = mul(meshData.normalMatrix, float4(IN.normal, 1.f)).xyz;
	result.fragPos = mul(meshData.modelMatrix, float4(IN.position, 1.f));
	result.position = mul(sceneData.viewProjection, result.fragPos);
	
	return result;
}
