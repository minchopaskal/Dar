#include "common.hlsli"

float4 main(PSInput IN) : SV_TARGET
{
	return IN.uv.x < 0.5 ? float4(IN.uv, 0.f, 1.f) : float4(0.f, IN.uv, 1.f);
}