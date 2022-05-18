struct PSInput {
	float2 uv : TEXCOORD;
};

SamplerState Sampler : register(s0, space0);

float4 main(PSInput IN) : SV_TARGET {
	Texture2D<float4> CudaRenderTexture = ResourceDescriptorHeap[0];
	return CudaRenderTexture.Sample(Sampler, IN.uv);
}