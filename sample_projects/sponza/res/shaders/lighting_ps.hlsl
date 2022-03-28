#include "lighting_common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint DEPTH_BUFFER_INDEX = 1;

static const uint GBUFFER_INDEX_OFFSET = 2;

static const uint GBUFFER_ALBEDO_INDEX = 0;
static const uint GBUFFER_NORMALS_INDEX = 1;
static const uint GBUFFER_MRO_INDEX = 2;
static const uint GBUFFER_POSITION_INDEX = 3;

float4 main(PSInput IN) : SV_TARGET {
	Material material;

	material.albedo = getColorFromTexture(GBUFFER_ALBEDO_INDEX, GBUFFER_INDEX_OFFSET, IN.uv, float4(0.f, 0.f, 0.f, 0.f));

	if (material.albedo.w == 0.f) {
		discard;
		return 0.f;
	}

	material.normal = getColorFromTexture(GBUFFER_NORMALS_INDEX, GBUFFER_INDEX_OFFSET, IN.uv).xyz;
	material.metalnessRoughnessOcclusion = getColorFromTexture(GBUFFER_MRO_INDEX, GBUFFER_INDEX_OFFSET, IN.uv).rgb;
	material.position = getColorFromTexture(GBUFFER_POSITION_INDEX, GBUFFER_INDEX_OFFSET, IN.uv).xyz;
	

	if (sceneData.showGBuffer == 1) {
		return material.albedo;
	} else if (sceneData.showGBuffer == 2) {
		return float4(material.normal, 1.f);
	} else if (sceneData.showGBuffer == 3) {
		const float c = material.metalnessRoughnessOcclusion.r;
		return float4(c, c, c, 1.f);
	} else if (sceneData.showGBuffer == 4) {
		const float c = material.metalnessRoughnessOcclusion.g;
		return float4(c, c, c, 1.f);
	} else if (sceneData.showGBuffer == 5) {
		const float c = material.metalnessRoughnessOcclusion.b;
		return float4(c, c, c, 1.f);
	} else if (sceneData.showGBuffer == 6) {
		return float4(normalize(material.position), 1.f);
	}

	return evalLights(material, LIGHTS_BUFFER_INDEX);
}