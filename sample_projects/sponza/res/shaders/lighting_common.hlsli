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

struct Material {
	float4 albedo;
	float3 metalnessRoughnessOcclusion;
	float3 normal;
	float3 position;
};

LightColors evalBlinnPhong(LightData light, float3 lightDir, Material material) {
	// Should be taken from the material, but we will support PBR materials,
	// so this code is for debugging purposes only.
	const int shininess = 16;

	const float3 invLightDir = -lightDir;
	const float3 viewDir = normalize(sceneData.cameraPosition.xyz - material.position);

	// Phong model calculations
	/*
	const float3 reflectDir = reflect(lightDir, material.normal);
	const float specularIntensity = pow(max(dot(viewDir, reflectDir), 0.f), shininess);
	*/

	// Blinn-Phong
	const float3 halfwayVector = normalize(viewDir + invLightDir);
	const float specularIntensity = pow(max(dot(material.normal, halfwayVector), 0.f), shininess);

	const float lightIntensity = max(dot(material.normal, invLightDir), 0.f);

	LightColors result;
	result.diffuse = lightIntensity * light.diffuse * material.albedo.xyz;
	result.ambient = light.ambient * material.albedo.xyz;
	result.specular = specularIntensity * light.specular * material.metalnessRoughnessOcclusion.r;

	return result;
}

float4 evalLights(Material material, const uint lightsBufferIndex) {
	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[lightsBufferIndex];

	LightColors lightColors = {
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 0.f)
	};

	for (int i = 0; i < sceneData.numLights; ++i) {
		const LightData light = lights[i];
		const float lightWeight = 1.f / (sceneData.numLights);

		if (light.type == LightType::Point) {
			const float3 lightDir = normalize(material.position - light.position);
			LightColors pointLight = evalBlinnPhong(light, lightDir, material);

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
			LightColors directionLight = evalBlinnPhong(light, lightDir, material);

			lightColors.diffuse += lightWeight * directionLight.diffuse;
			lightColors.specular += lightWeight * directionLight.specular;
			lightColors.ambient += lightWeight * directionLight.ambient;
		}

		if (light.type == LightType::Spot) {
			//const float3 lightDir = normalize(IN.fragPos.xyz - light.position);
			//const float3 spotDir = light.direction;
			const float3 lightDir = normalize(material.position - sceneData.cameraPosition.xyz);
			const float3 spotDir = sceneData.cameraDir.xyz;
			const float theta = dot(lightDir, spotDir);

			if (theta > light.outerCutoff) {
				LightColors spotLight = evalBlinnPhong(light, lightDir, material);

				if (theta < light.innerCutoff) {
					const float spotEdgeIntensity = (theta - light.outerCutoff) / (light.innerCutoff - light.outerCutoff);

					spotLight.diffuse *= spotEdgeIntensity;
					spotLight.specular *= spotEdgeIntensity;
				}

				lightColors.diffuse += lightWeight * spotLight.diffuse;
				lightColors.specular += lightWeight * spotLight.specular;
				lightColors.ambient += lightWeight * spotLight.ambient;
			} else {
				lightColors.ambient += lightWeight * light.ambient * material.albedo.xyz;
			}
		}
	}

	return float4(lightColors.diffuse + lightColors.ambient + lightColors.specular, 1.f);
}