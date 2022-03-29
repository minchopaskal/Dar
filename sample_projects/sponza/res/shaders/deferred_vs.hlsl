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
	float4 fragPos : POSITION0;
	float3 normal : NORMAL;
	nointerpolation float3x3 TBN : TANGENT_MATRIX; // tangent space -> world space
	float2 uv : TEXCOORD;
};

VSOutput main(VSInput IN)
{
	VSOutput result;
	
	result.fragPos = mul(meshData.modelMatrix, float4(IN.position, 1.f));
	result.position = mul(sceneData.viewProjection, result.fragPos);

	const float3 N = normalize(mul(meshData.normalMatrix, float4(IN.normal, 0.f)).xyz);

	float3 T = normalize(mul(meshData.normalMatrix, float4(IN.tangent, 0.f)).xyz);
	T = normalize(T - N * dot(T, N));
	const float3 B = normalize(cross(N, T));

	result.TBN = float3x3(T, B, N);

	result.normal = N;
	result.uv = IN.uv;

	return result;
}
