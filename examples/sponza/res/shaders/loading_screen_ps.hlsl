#include "loading_screen_common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
	float4 position : SV_Position;
};

ConstantBuffer<LoadingScreenConstData> constData : register(b0);

float4 main(PSInput IN) : SV_TARGET {
	float aspRatio = float(constData.width) / float(constData.height);
	float2 st = (IN.uv * 2. - 1.) * float2(1.f, 1. / aspRatio);
	
	float dst = smoothstep(0.1, 0.6, distance(st, float2(.0, .0)));
	
	return (1.0 - dst) * float4(abs(sin(constData.time * 3.)) * IN.uv.x, abs(cos(constData.time * 2.)) * IN.uv.y, 0.f, 1.f);
}
