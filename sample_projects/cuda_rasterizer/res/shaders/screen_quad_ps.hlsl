struct PSInput {
	float2 uv : TEXCOORD;
};

SamplerState Sampler : register(s0, space0);
Texture2D CudaRenderTexture : register(t0, space0);

float4 main(PSInput IN) : SV_TARGET{
	return CudaRenderTexture.Sample(Sampler, IN.uv);
}