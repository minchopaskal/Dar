#include "common.hlsli"

struct PSInput {
	float4 position : SV_Position;
	float2 uv: TEXCOORD;
};

static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint MATERIALS_BUFFER_INDEX = LIGHTS_BUFFER_INDEX + 1;
static const uint TEXTURE_BUFFERS_START = MATERIALS_BUFFER_INDEX + 1;

void main(PSInput IN) : SV_TARGET {
	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialData material = materials[meshData.materialId];
	float4 albedo = getColorFromTexture(material.baseColorIndex, TEXTURE_BUFFERS_START, IN.uv, TextureUsage::Default, float4(0.f, 0.f, 0.f, 1.f));

	clip(albedo.a - 0.1f);
}