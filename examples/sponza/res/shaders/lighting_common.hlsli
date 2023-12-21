#ifndef LIGHTING_COMMON_HLSLI
#define LIGHTING_COMMON_HLSLI

#include "common.hlsli"

static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint DEPTH_BUFFER_INDEX = 1;

static const uint SHADOW_MAP_BUFFERS_START_INDEX = 2;

static const uint GBUFFER_INDEX_OFFSET = SHADOW_MAP_BUFFERS_START_INDEX + MAX_SHADOW_MAPS_COUNT;

static const uint GBUFFER_ALBEDO_INDEX = 0;
static const uint GBUFFER_NORMALS_INDEX = 1;
static const uint GBUFFER_MRO_INDEX = 2;
static const uint GBUFFER_POSITION_INDEX = 3;

struct LightColors {
	float3 diffuse;
	float3 specular;
	float3 ambient;
};

struct Material {
	float4 albedo;
	float3 metalnessRoughnessOcclusion;
	float3 normal;
	float3 position;
};

LightColors evalBlinnPhong(LightData light, float3 lightDir, float3 viewDir, Material material) {
	// Should be taken from the material, but we will support PBR materials,
	// so this code is for debugging purposes only.
	const int shininess = 16;

	// Phong model calculations
	/*
	const float3 reflectDir = reflect(lightDir, material.normal);
	const float specularIntensity = pow(max(dot(viewDir, reflectDir), 0.f), shininess);
	*/

	// Blinn-Phong
	const float3 halfwayVector = normalize(viewDir + lightDir);
	const float specularIntensity = pow(max(dot(material.normal, halfwayVector), 0.f), shininess);

	const float lightIntensity = max(dot(material.normal, lightDir), 0.f);

	LightColors result;
	result.diffuse = lightIntensity * light.diffuse * material.albedo.xyz;
	result.ambient = light.ambient * material.albedo.xyz;
	result.specular = specularIntensity * light.specular * material.metalnessRoughnessOcclusion.r;

	return result;
}

float3 fresnelSchlick(const float3 F0, float3 V, float3 H) {
	return F0 + (1.f - F0) * pow(clamp(1.f - dot(H, V), 0.f, 1.f), 5);
}

float ndfTrowbridgeReitzGGX(float NoH, float roughness) {
	const float a = roughness * roughness;
	const float a2 = a * a;

	float denom = NoH * NoH * (a2 - 1) + 1;
	denom = PI * denom * denom;

	return a2 / denom;
}

// Note that optimize the brdf calculations by cancelling the numerator of the geometry GGX
// with the denominator of the reflectance BRDF
float geometrySchlick_SmithGGX(float3 N, float3 V, float3 L, float roughness) {
	const float k = (roughness*roughness) / 2.f;

	/*float r = (roughness + 1.0);
	float k = (r * r) / 8.0;*/

	const float NoV = max(0.f, dot(N, V));
	const float NoL = max(0.f, dot(N, L));

	const float ggxV = (NoV * (1 - k) + k);
	const float ggxL = (NoL * (1 - k) + k);

	return rcp(max(ggxV * ggxL, 1e-7f));
}

float3 evalOutputRadiance(Material material, float3 lightColor, float3 N, float3 V, float3 L, float attenuation, float roughness) {
	const float metalness = material.metalnessRoughnessOcclusion.r;
	float3 F0 = 0.04f;
	F0 = lerp(F0, material.albedo.rgb, metalness);

	const float3 H = normalize(L + V);
	const float NoH = max(0.f, dot(N, H));

	const float3 kS = fresnelSchlick(F0, V, H);
	const float D = ndfTrowbridgeReitzGGX(NoH, roughness);
	const float G = geometrySchlick_SmithGGX(N, V, L, roughness);
	//         D * G * F
	// brdf = -----------
	//         4*NoL*NoV
	// But the visibility GGX contains NoL*NoV as a numerator so we cancel that out.
	const float3 specular = kS * D * G * 0.25;

	const float3 kD = (1.f - kS);
	const float3 diffuse = (kD * material.albedo.rgb) / PI;
	const float3 brdf = diffuse + specular;

	const float NoL = max(0.f, dot(N, L));
	const float3 radiance = lightColor * attenuation;
	return brdf * radiance * NoL;
}

float linearizeDepth(float depth, float zNear = -9999.f, float zFar = -9999.f) {
	zNear = max(zNear, sceneData.nearPlane);
	zFar  = max(zFar, sceneData.farPlane);
	// TODO: fix me
	return zNear / (zFar - depth * (zFar - zNear));
}

float calcShadowFactor(Material material, LightData light) {
	Texture2D<float> shadowMap = ResourceDescriptorHeap[SHADOW_MAP_BUFFERS_START_INDEX + light.shadowMapIndexOffset];

	uint width, height, numMips;
	shadowMap.GetDimensions(0, width, height, numMips);

	// Assuming square shadow map here
	float dx = 1.0f / (float)width;
	const float2 offsets[9] =
	{
		float2(-dx,  -dx), float2(0.0f,  -dx), float2(dx,  -dx),
		float2(-dx, 0.0f), float2(0.0f, 0.0f), float2(dx, 0.0f),
		float2(-dx,  +dx), float2(0.0f,  +dx), float2(dx,  +dx)
	};

	float4 shadowPosClip = mul(light.viewProjection, float4(material.position, 1.f));
	float3 shadowPosNDC = shadowPosClip.xyz / shadowPosClip.w;
	float3 uvDepth = float3(shadowPosNDC.xy * 0.5 + 0.5, shadowPosNDC.z);
	// need to invert y as from top to bottom:
	// NDC y goes from 1 to -1, where as in tex-space it goes from 0 to 1
	uvDepth.y = 1. - uvDepth.y;

	float percentLit = 0.0f;
	[unroll]
	for (int i = 0; i < 9; ++i) {
		percentLit += shadowMap.SampleCmp(SamplerDepth, uvDepth.xy + offsets[i], uvDepth.z);
	}

	return percentLit / 9.0;
}

float luminance(float3 color) {
	return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

float4 evalLights(Material material) {
	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[LIGHTS_BUFFER_INDEX];

	const float3 V = normalize(sceneData.cameraPosition.xyz - material.position);
	const float3 N = material.normal; // should be normalized already!
	float metalness = material.metalnessRoughnessOcclusion.b;
	float roughness = material.metalnessRoughnessOcclusion.g;
	float occlusion = material.metalnessRoughnessOcclusion.r;

	float3 lighting = 0.f;

	for (int i = 0; i < sceneData.numLights; ++i) {
		const LightData light = lights[i];

		float shadowFactor = 1.f;
		if (light.shadowMapIndexOffset >= 0 && light.shadowMapIndexOffset < MAX_SHADOW_MAPS_COUNT) {
			shadowFactor = calcShadowFactor(material, light);
		}

		if (light.type == LightType::Point) {
			const float3 L = normalize(light.position - material.position);
		
			const float distance = length(light.position - material.position);
			const float c = light.attenuation.x;
			const float l = light.attenuation.y;
			const float q = light.attenuation.z;
			//const float attenuation = 1.f / (distance * distance); // physically correct, but the other method gives us more control
			const float attenuation = 1.f / (c + l * distance + q * distance * distance);
			lighting += shadowFactor * evalOutputRadiance(material, light.diffuse, N, V, L, attenuation, roughness);
		}

		if (light.type == LightType::Directional) {
			const float3 L = -normalize(light.direction);
			lighting += shadowFactor * evalOutputRadiance(material, light.diffuse, N, V, L, 1.f, roughness);
		}
		
		if (light.type == LightType::Spot && sceneData.spotLightON) {
			const float3 lightDir = normalize(light.position - material.position);
			const float3 spotDir = -normalize(light.direction);
			const float theta = dot(lightDir, spotDir);
			const float3 L = lightDir;

			if (theta > light.outerAngleCutoff) {
				float3 spotLighting = evalOutputRadiance(material, light.diffuse, N, V, L, 1.f, roughness);
		
				if (theta < light.innerAngleCutoff) {
					const float spotEdgeIntensity = (theta - light.outerAngleCutoff) / (light.innerAngleCutoff - light.outerAngleCutoff);
		
					spotLighting *= spotEdgeIntensity;
				}
				
				// We need to linearize here as spotlight uses perspective projection.
				shadowFactor = linearizeDepth(shadowFactor, light.zNear, light.zFar);
				lighting += shadowFactor * spotLighting;
			}
		}
	}

	const float3 ambient = 0.01 * material.albedo.rgb * material.metalnessRoughnessOcclusion.b;
	float3 color = lighting+ambient;

	return float4(color, 1.f);
}

#endif // LIGHTING_COMMON_HLSLI
