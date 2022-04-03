#ifndef LIGHTING_COMMON_HLSLI
#define LIGHTING_COMMON_HLSLI

#include "common.hlsli"

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

	return rcp(ggxV * ggxL);
}

float3 evalOutputRadiance(Material material, float3 lightColor, float3 N, float3 V, float3 L, float attenuation, float roughness) {
	const float metalness = material.metalnessRoughnessOcclusion.r;
	float3 F0 = 0.04f;
	F0 = lerp(F0, material.albedo.rgb, metalness);

	const float3 H = normalize(L + V);
	const float NoH = max(0.f, dot(N, H));
	const float3 kS = fresnelSchlick(F0, V, H);
	//         D * G * F
	// brdf = -----------
	//         4*NoL*NoV
	// But the visibility GGX contains NoL*NoV as a numerator so we cancel that out.
	const float3 specular = kS * ndfTrowbridgeReitzGGX(NoH, roughness) * geometrySchlick_SmithGGX(N, V, L, roughness) * 0.25;
	
	const float3 kD = (1.f - kS);
	const float3 diffuse = (kD * material.albedo.rgb) / PI;
	const float3 brdf = diffuse + specular;

	const float NoL = max(0.f, dot(N, L));
	const float3 radiance = lightColor * attenuation;
	return brdf * radiance * NoL;
}

float luminance(float3 color) {
	return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

float4 evalLights(Material material, const uint lightsBufferIndex) {
	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[lightsBufferIndex];

	const float3 V = normalize(sceneData.cameraPosition.xyz - material.position);
	const float3 N = material.normal; // should be normalized already!
	float metalness = material.metalnessRoughnessOcclusion.b;
	float roughness = material.metalnessRoughnessOcclusion.g;
	float occlusion = material.metalnessRoughnessOcclusion.r;

	float3 lighting = 0.f;

	for (int i = 0; i < sceneData.numLights; ++i) {
		const LightData light = lights[i];

		if (light.type == LightType::Point) {
			const float3 L = normalize(light.position - material.position);

			const float distance = length(light.position - material.position);
			const float c = light.attenuation.x;
			const float l = light.attenuation.y;
			const float q = light.attenuation.z;
			//const float attenuation = 1.f / (distance * distance); // physically correct, but the other method gives us more control
			const float attenuation = 1.f / (c + l * distance + q * distance * distance);
			lighting += evalOutputRadiance(material, light.diffuse, N, V, L, attenuation, roughness);
		}

		if (light.type == LightType::Directional) {
			const float3 L = -normalize(light.direction);
			lighting += evalOutputRadiance(material, light.diffuse, N, V, L, 1.f, roughness);
		}

		if (light.type == LightType::Spot && sceneData.spotLightON) {
			//const float3 lightDir = normalize(IN.fragPos.xyz - light.position);
			//const float3 spotDir = light.direction;
			const float3 L = normalize(V);
			const float3 spotDir = -sceneData.cameraDir.xyz;
			const float theta = dot(L, spotDir);

			if (theta > light.outerAngleCutoff) {
				float3 spotLighting = evalOutputRadiance(material, light.diffuse, N, V, L, 1.f, roughness);

				if (theta < light.innerAngleCutoff) {
					const float spotEdgeIntensity = (theta - light.outerAngleCutoff) / (light.innerAngleCutoff - light.outerAngleCutoff);

					spotLighting *= spotEdgeIntensity;
				}

				lighting += spotLighting;
			}
		}
	}

	const float3 ambient = 0.01 * material.albedo.rgb * material.metalnessRoughnessOcclusion.b;
	float3 color = lighting+ambient;

	return float4(color, 1.f);
}

#endif // LIGHTING_COMMON_HLSLI
