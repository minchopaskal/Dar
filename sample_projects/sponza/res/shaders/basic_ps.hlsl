struct SceneData {
	row_major matrix viewProjection;
	float4 cameraPos; // world-space
	float4 cameraDir; // world-space
};

struct MeshData {
	row_major matrix modelMatrix;
	row_major matrix normalMatrix;
	uint materialId;
};

struct MaterialData {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int normal;
};

struct PSInput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
};

struct PSOutput
{
	float4 diffuse;
	float4 specular;
	float4 normal;
	float4 position;
};

SamplerState Sampler : register(s0);
ConstantBuffer<SceneData> sceneData : register(b0);
ConstantBuffer<MeshData> meshData : register(b2);

// TODO: This should go in a common file.
static const uint MATERIALS_BUFFER_INDEX = 0;
static const uint TEXTURE_BUFFERS_START = 1;
static const uint INVALID_MATERIAL_INDEX = 0xffffffff;

float4 getColorFromTex(uint textureIndex, float2 uv, float4 defaultColor = float4(1.f, 1.f, 1.f, 1.f)) {
	float4 color = defaultColor;
	if (textureIndex != INVALID_MATERIAL_INDEX) {
		Texture2D<float4> tex = ResourceDescriptorHeap[textureIndex + TEXTURE_BUFFERS_START];
		color = tex.Sample(Sampler, uv);
	}

	return color;
}

PSOutput main(PSInput IN) : SV_Target
{
	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialData material = materials[meshData.materialId];

	PSOutput output;

	output.diffuse = getColorFromTex(material.diffuse, IN.uv, float4(0.f, 0.f, 0.f, 1.f));
	output.specular = getColorFromTex(material.specular, IN.uv, float4(.5f, .5f, .5f, 1.f));
	// TODO: normal mapping.
	output.normal = float4(IN.normal, 0.f);//getColorFromTex(material.normal, IN.uv, float4(IN.normal, 0.f));
	output.normal = float4(normalize(output.normal.xyz), 0.f);
	output.position = IN.fragPos;

	return output;
}