struct MaterialData {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int normal;
};

struct MeshData {
	row_major matrix modelMatrix;
	row_major matrix normalMatrix;
	uint materialId;
};

struct PSInput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
};

SamplerState Sampler : register(s0);
ConstantBuffer<MeshData> meshData : register(b2);

float4 main(PSInput IN) : SV_TARGET
{
	// This should go in a common file.
	const uint LIGHTS_BUFFER_INDEX = 0;
	const uint MATERIALS_BUFFER_INDEX = 1;
	const uint TEXTURE_BUFFERS_START = 2;

	const float3 lightDir = normalize(float3(-1.f, 1.f, 0.f));

	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	MaterialData material = materials[meshData.materialId];

	Texture2D<float4> diffuseTex = ResourceDescriptorHeap[material.diffuse + TEXTURE_BUFFERS_START];
	const float4 diffuse = diffuseTex.Sample(Sampler, IN.uv);

	if (diffuse.w < 1e-6f) {
		return float4(0.f, 0.f, 0.f, 0.f);
	}
	const float lightIntensity = dot(lightDir, IN.normal);

	return lightIntensity * diffuse;
}