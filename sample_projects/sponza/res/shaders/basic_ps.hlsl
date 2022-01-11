struct MaterialData {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int normal;
};

struct PSInput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
};

SamplerState Sampler : register(s0);
ConstantBuffer<MaterialData> material : register(b1);

float4 main(PSInput IN) : SV_TARGET
{
	const float3 lightDir = normalize(float3(-1.f, 1.f, 0.f));

	Texture2D<float4> diffuseTex = ResourceDescriptorHeap[material.diffuse];
	const float4 diffuse = diffuseTex.Sample(Sampler, IN.uv);
	const float lightIntensity = dot(lightDir, IN.normal);

	return lightIntensity * diffuse;
}