struct MaterialData {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int normal;
};

struct PSInput
{
	float2 uv : TEXCOORD;
};

SamplerState Sampler : register(s0);
ConstantBuffer<MaterialData> materials : register(b0);

float4 main(PSInput IN) : SV_TARGET
{
	Texture2D<float4> tex = ResourceDescriptorHeap[0];
	return tex.Sample(Sampler, IN.uv);
}