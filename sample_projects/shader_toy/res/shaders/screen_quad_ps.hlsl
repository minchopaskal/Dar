struct PSInput {
	float2 uv : TEXCOORD;
};

float4 main(PSInput IN) : SV_TARGET
{
	return float4(IN.uv, 0.f, 1.f);
}