#include "lighting_common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint GBUFFER_DIFFUSE_INDEX = 1;
static const uint GBUFFER_SPECULAR_INDEX = 2;
static const uint GBUFFER_NORMAL_INDEX = 3;
static const uint GBUFFER_POSITION_INDEX = 4;

float4 main(PSInput IN) : SV_TARGET {

	MaterialData material;

	material.diffuse = getColorFromTexture(GBUFFER_DIFFUSE_INDEX, IN.uv, float4(0.f, 0.f, 0.f, 1.f));
	material.specular = getColorFromTexture(GBUFFER_SPECULAR_INDEX, IN.uv, float4(.5f, .5f, .5f, 1.f));
	material.normal = getColorFromTexture(GBUFFER_NORMAL_INDEX, IN.uv).xyz;
	material.position = getColorFromTexture(GBUFFER_POSITION_INDEX, IN.uv).xyz;

	if (sceneData.showGBuffer == 1) {
		return material.diffuse;
	} else if (sceneData.showGBuffer == 2) {
		return material.specular;
	} else if (sceneData.showGBuffer == 3) {
		return float4(material.normal, 1.f);
	} else if (sceneData.showGBuffer == 4) {
		return float4(normalize(material.position), 1.f);
	}

	return evalLights(material, LIGHTS_BUFFER_INDEX);
}