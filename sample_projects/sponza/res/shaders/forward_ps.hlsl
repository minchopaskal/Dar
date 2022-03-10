#include "lighting_common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
};

static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint MATERIALS_BUFFER_INDEX = 1;
static const uint TEXTURE_BUFFERS_START = 2;

float4 main(PSInput IN) : SV_TARGET {
	StructuredBuffer<MaterialIndices> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialIndices materialIndices = materials[meshData.materialId];

	MaterialData material;
	material.diffuse = getColorFromTexture(materialIndices.diffuseIndex + TEXTURE_BUFFERS_START, IN.uv, float4(0.f, 0.f, 0.f, 1.f));
	if (material.diffuse.w < 1e-6) {
		discard;
	}
	material.specular = getColorFromTexture(materialIndices.specularIndex + TEXTURE_BUFFERS_START, IN.uv, float4(.5f, .5f, .5f, 1.f));
	material.normal = IN.normal;// getColorFromTexture(materialIndices.normalIndex + TEXTURE_BUFFERS_START, IN.uv, float4(IN.normal, 1.f)).xyz;
	material.position = IN.fragPos.xyz;

	return evalLights(material, LIGHTS_BUFFER_INDEX);
}