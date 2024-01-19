#include "common.hlsli"

struct VSInput
{
	float3 position : POSITION0;
	float3 normal : NORMAL;
	float3 tangent : TANGENT;
	float2 uv : TEXCOORD;
	float4 weights : JOINTWEIGHTS;
	uint indices :   JOINTINDICES;
};

struct VSOutput
{
	float4 position : SV_Position;
	float4 color: COLOR;
	float3 normal : NORMAL;
	nointerpolation float3x3 TBN : TANGENT_MATRIX; // tangent space -> world space
	float2 uv : TEXCOORD;
};

float getComp(float4 v, int comp) {
	if (comp == 0) {
		return v.x;
	}

	if (comp == 1) {
		return v.y;
	}

	if (comp == 2) {
		return v.z;
	}

	return v.w;
}

VSOutput main(VSInput IN)
{
	VSOutput result;
	
	float4 worldPos = mul(meshData.modelMatrix, float4(IN.position, 1.f));
	result.position = mul(sceneData.viewProjection, worldPos);

	result.color = float4(0.f, 0.f, 1.f, 1.f);
	for (int i = 0; i < 4; ++i) {
		if ( ((IN.indices >> (8 * i)) & 0xFF) == sceneData.boneIdx) {
			if (getComp(IN.weights, i) > 0.4) {
				result.color = float4(1.f, 0.f, 0.f, 1.f);
			} else if (getComp(IN.weights, i) > 0.3) {
				result.color = float4(0.f, 1.f, 0.f, 1.f);
			} else if (getComp(IN.weights, i) > 0.2) {
				result.color = float4(1.f, 1.f, 0.f, 1.f);
			}
		}
	}

	const float3 N = normalize(mul(meshData.normalMatrix, float4(IN.normal, 0.f)).xyz);
	result.normal = N;

	float3 T = normalize(mul(meshData.normalMatrix, float4(IN.tangent, 0.f)).xyz);
	T = normalize(T - N * dot(T, N));
	const float3 B = normalize(cross(N, T));
	result.TBN = float3x3(T, B, N);

	result.uv = IN.uv;

	return result;
}
