struct MaterialData {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int normal;
};

enum LightType : int {
	Invalid = -1,

	Point = 0,
	Directional,
	Spot,

	Count
};

struct LightData {
	float3 position;
	float3 diffuse;
	float3 ambient;
	float3 specular;
	float3 attenuation;
	float3 direction;
	float innerAngleCutoff;
	float outerAngleCutoff;
	LightType type;
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
	float4 fragPos : POSITION0;
	float3 cameraPos : POSITION1;
};

SamplerState Sampler : register(s0);
ConstantBuffer<MeshData> meshData : register(b2);

float4 main(PSInput IN) : SV_TARGET
{
	// This should go in a common file.
	const uint LIGHTS_BUFFER_INDEX = 0;
	const uint MATERIALS_BUFFER_INDEX = 1;
	const uint TEXTURE_BUFFERS_START = 2;

	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[LIGHTS_BUFFER_INDEX];
	const LightData dirLight = lights[0];

	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialData material = materials[meshData.materialId];

	Texture2D<float4> diffuseTex = ResourceDescriptorHeap[material.diffuse + TEXTURE_BUFFERS_START];
	const float4 diffuseColor = diffuseTex.Sample(Sampler, IN.uv);

	if (diffuseColor.w < 1e-6f) {
		return float4(0.f, 0.f, 0.f, 0.f);
	}

	const float3 viewDir = normalize(IN.cameraPos - IN.fragPos.xyz);
	const float3 reflectDir = reflect(dirLight.direction, IN.normal);
	float specularIntensity = 0.5f * pow(max(dot(viewDir, reflectDir), 0.f), 8);

	const float lightIntensity = dot(dirLight.direction, IN.normal);
	const float4 diffuse = float4(lightIntensity * dirLight.diffuse, 1.f);
	const float4 ambient = float4(dirLight.ambient, 1.f);
	const float4 specular = float4(specularIntensity * dirLight.specular, 1.f);

	return (diffuse + ambient + specular) * diffuseColor;
}