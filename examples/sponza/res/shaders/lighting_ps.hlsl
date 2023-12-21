#include "lighting_common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

float3 calcPositionFromDepth(float2 uv, float depth) {
	float4 clipSpace = float4(uv * 2.f - 1.f, depth, 1.f);
	clipSpace.y *= -1.f;
	
	float4 viewSpace = mul(sceneData.invProjection, clipSpace);
	viewSpace /= viewSpace.w;

	float4 worldSpace = mul(sceneData.invView, viewSpace);

	return worldSpace.xyz;
}

float4 main(PSInput IN) : SV_TARGET {
	if (sceneData.showGBuffer == 8) {
		Texture2D<float> shadowMap = ResourceDescriptorHeap[SHADOW_MAP_BUFFERS_START_INDEX + 0];
		float shadowMapDepth = shadowMap.Sample(Sampler, IN.uv);
		return float4(shadowMapDepth, 0.f, 0.f, 1.f);
	}
	if (sceneData.showGBuffer == 9) {
		Texture2D<float> shadowMap = ResourceDescriptorHeap[SHADOW_MAP_BUFFERS_START_INDEX + 1];
		float shadowMapDepth = linearizeDepth(shadowMap.Sample(Sampler, IN.uv));
		return float4(shadowMapDepth, 0.f, 0.f, 1.f);
	}

	Material material;
	material.albedo = getColorFromTexture(GBUFFER_ALBEDO_INDEX, GBUFFER_INDEX_OFFSET, IN.uv, TextureUsage::Default, float4(0.f, 0.f, 0.f, 0.f));

	material.normal = getColorFromTexture(GBUFFER_NORMALS_INDEX, GBUFFER_INDEX_OFFSET, IN.uv).xyz;
	material.metalnessRoughnessOcclusion = getColorFromTexture(GBUFFER_MRO_INDEX, GBUFFER_INDEX_OFFSET, IN.uv).rgb;

	Texture2D<float> depthMap = ResourceDescriptorHeap[DEPTH_BUFFER_INDEX];
	float depth = depthMap.Sample(Sampler, IN.uv);
	material.position = calcPositionFromDepth(IN.uv, depth);

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
	} else if (sceneData.showGBuffer == 7) {
		return float4(linearizeDepth(depth), 0., 0., 1.);
	}

	return evalLights(material);
}