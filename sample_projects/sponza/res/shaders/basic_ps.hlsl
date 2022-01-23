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

struct PSInput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
};

SamplerState Sampler : register(s0);
ConstantBuffer<SceneData> sceneData : register(b0);
ConstantBuffer<MeshData> meshData : register(b2);

// TODO: This should go in a common file.
static const uint LIGHTS_BUFFER_INDEX = 0;
static const uint MATERIALS_BUFFER_INDEX = 1;
static const uint TEXTURE_BUFFERS_START = 2;
static const uint INVALID_MATERIAL_INDEX = 0xffffffff;

float4 getColorFromTex(uint textureIndex, float2 uv, float4 defaultColor = float4(1.f, 1.f, 1.f, 1.f)) {
	float4 color = defaultColor;
	if (textureIndex != INVALID_MATERIAL_INDEX) {
		Texture2D<float4> tex = ResourceDescriptorHeap[textureIndex + TEXTURE_BUFFERS_START];
		color = tex.Sample(Sampler, uv);
	}

	return color;
}

struct LightColors {
	float3 diffuse;
	float3 specular;
	float3 ambient;
};

struct MaterialColors {
	float4 diffuse;
	float4 specular;
	float4 normal;
};

LightColors getLightValues(LightData light, float3 lightDir, MaterialColors colors, PSInput IN) {
	const float3 viewDir = normalize(sceneData.cameraPos.xyz - IN.fragPos.xyz);
	const float3 reflectDir = reflect(lightDir, IN.normal);
	const float specularIntensity = 0.5f * pow(max(dot(viewDir, reflectDir), 0.f), 8);
	const float lightIntensity = max(dot(IN.normal, -lightDir), 0.f);
	
	LightColors result;
	result.diffuse = lightIntensity * light.diffuse * colors.diffuse.xyz;
	result.ambient = light.ambient * colors.diffuse.xyz;
	result.specular = specularIntensity * light.specular * colors.specular.xyz;

	return result;
}

float4 main(PSInput IN) : SV_TARGET
{
	StructuredBuffer<LightData> lights = ResourceDescriptorHeap[LIGHTS_BUFFER_INDEX];

	MaterialColors colors;
	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialData material = materials[meshData.materialId];

	colors.diffuse = getColorFromTex(material.diffuse, IN.uv, float4(0.f, 0.f, 0.f, 1.f));
	if (colors.diffuse.w < 1e-6) {
		discard;
	}

	colors.specular = getColorFromTex(material.specular, IN.uv, float4(.5f, .5f, .5f, 1.f));
	colors.normal = getColorFromTex(material.normal, IN.uv, float4(IN.normal, 1.f));

	uint numLights, temp;
	lights.GetDimensions(numLights, temp);
	LightColors lightColors = {
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 0.f),
		float3(0.f, 0.f, 0.f)
	};

	for (int i = 0; i < numLights; ++i) {
		const LightData light = lights[2];
		const float lightWeight = 1.f;// / numLights;
		
		if (light.type == Point) {
			const float3 lightDir = normalize(IN.fragPos.xyz - light.position);
			LightColors pointLight = getLightValues(light, lightDir, colors, IN);

			const float distance = length(light.position - IN.fragPos.xyz);
			const float c = light.attenuation.x;
			const float l = light.attenuation.y;
			const float q = light.attenuation.z;
			const float attenuation = 1.f / (c + l * distance + q * distance * distance);
			pointLight.diffuse *= attenuation;
			pointLight.specular *= attenuation;
			pointLight.ambient *= attenuation;

			lightColors.diffuse  += lightWeight * pointLight.diffuse;
			lightColors.specular += lightWeight * pointLight.specular;
			lightColors.ambient  += lightWeight * pointLight.ambient;
		}

		if (light.type == Directional) {
			const float3 lightDir = light.direction;
			LightColors directionLight = getLightValues(light, lightDir, colors, IN);

			lightColors.diffuse  += lightWeight * directionLight.diffuse;
			lightColors.specular += lightWeight * directionLight.specular;
			lightColors.ambient  += lightWeight * directionLight.ambient;
		}

		if (light.type == Spot) {
			//const float3 lightDir = normalize(IN.fragPos.xyz - light.position);
			//const float3 spotDir = light.direction;
			const float3 lightDir = normalize(IN.fragPos.xyz - sceneData.cameraPos.xyz);
			const float3 spotDir = sceneData.cameraDir.xyz;
			const float theta = dot(lightDir, spotDir);

			if (theta > light.outerCutoff) {
				LightColors spotLight = getLightValues(light, lightDir, colors, IN);

				if (theta < light.innerCutoff) {
					const float spotEdgeIntensity = (theta - light.outerCutoff) / (light.innerCutoff - light.outerCutoff);

					spotLight.diffuse *= spotEdgeIntensity;
					spotLight.specular *= spotEdgeIntensity;
				}

				lightColors.diffuse  += lightWeight * spotLight.diffuse;
				lightColors.specular += lightWeight * spotLight.specular;
				lightColors.ambient  += lightWeight * spotLight.ambient;
			} else {
				lightColors.ambient += lightWeight * light.ambient * colors.diffuse.xyz;
			}
		}
	}

	return float4(lightColors.diffuse + lightColors.ambient + lightColors.specular, 1.f);
}