#include "common.hlsli"

enum LightType {
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
	float innerCutoff;
	float outerCutoff;
	LightType type;
};

struct LightColors {
	float3 diffuse;
	float3 specular;
	float3 ambient;
};

struct MaterialData {
	float4 diffuse;
	float4 specular;
	float3 position;
	float3 normal;
};

LightColors getLightValues(LightData light, float3 lightDir, MaterialData material) {
	const float3 viewDir = normalize(sceneData.cameraPos.xyz - material.position);
	const float3 reflectDir = reflect(lightDir, material.normal);
	const float specularIntensity = 0.5f * pow(max(dot(viewDir, reflectDir), 0.f), 8);
	const float lightIntensity = max(dot(material.normal, -lightDir), 0.f);

	LightColors result;
	result.diffuse = lightIntensity * light.diffuse * material.diffuse.xyz;
	result.ambient = light.ambient * material.diffuse.xyz;
	result.specular = specularIntensity * light.specular * material.specular.xyz;

	return result;
}

float4 evalLights(MaterialData material, const uint lightsBufferIndex) {
	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[lightsBufferIndex];

	LightColors lightColors = {
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 0.f)
	};

	for (int i = 0; i < sceneData.numLights; ++i) {
		const LightData light = lights[i];
		const float lightWeight = 1.f / sceneData.numLights;

		if (light.type == LightType::Point) {
			const float3 lightDir = normalize(material.position - light.position);
			LightColors pointLight = getLightValues(light, lightDir, material);

			const float distance = length(light.position - material.position);
			const float c = light.attenuation.x;
			const float l = light.attenuation.y;
			const float q = light.attenuation.z;
			const float attenuation = 1.f / (c + l * distance + q * distance * distance);
			pointLight.diffuse *= attenuation;
			pointLight.specular *= attenuation;
			pointLight.ambient *= attenuation;

			lightColors.diffuse += lightWeight * pointLight.diffuse;
			lightColors.specular += lightWeight * pointLight.specular;
			lightColors.ambient += lightWeight * pointLight.ambient;
		}

		if (light.type == LightType::Directional) {
			const float3 lightDir = light.direction;
			LightColors directionLight = getLightValues(light, lightDir, material);

			lightColors.diffuse += lightWeight * directionLight.diffuse;
			lightColors.specular += lightWeight * directionLight.specular;
			lightColors.ambient += lightWeight * directionLight.ambient;
		}

		if (light.type == LightType::Spot) {
			//const float3 lightDir = normalize(IN.fragPos.xyz - light.position);
			//const float3 spotDir = light.direction;
			const float3 lightDir = normalize(material.position - sceneData.cameraPos.xyz);
			const float3 spotDir = sceneData.cameraDir.xyz;
			const float theta = dot(lightDir, spotDir);

			if (theta > light.outerCutoff) {
				LightColors spotLight = getLightValues(light, lightDir, material);

				if (theta < light.innerCutoff) {
					const float spotEdgeIntensity = (theta - light.outerCutoff) / (light.innerCutoff - light.outerCutoff);

					spotLight.diffuse *= spotEdgeIntensity;
					spotLight.specular *= spotEdgeIntensity;
				}

				lightColors.diffuse += lightWeight * spotLight.diffuse;
				lightColors.specular += lightWeight * spotLight.specular;
				lightColors.ambient += lightWeight * spotLight.ambient;
			} else {
				lightColors.ambient += lightWeight * light.ambient * material.diffuse.xyz;
			}
		}
	}

	return float4(lightColors.diffuse + lightColors.ambient + lightColors.specular, 1.f);
}