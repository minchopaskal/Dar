#include "common.hlsli"

struct PSInput
{
	float4 position : SV_Position;
	float3 normal : NORMAL;
	row_major float3x3 TBN : TANGENT_MATRIX; // tangent space -> world space
	float2 uv : TEXCOORD;
};

struct PSOutput
{
	float4 albedo;
	float4 normal;
	float4 metalnessRoughnessOcclusion;
};

static const uint INVALID_TEXTURE_INDEX = uint(-1);
static const uint MATERIALS_BUFFER_INDEX = 0;
static const uint TEXTURE_BUFFERS_START = 1;

PSOutput main(PSInput IN) : SV_Target
{
	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialData material = materials[meshData.materialId];

	PSOutput output;

	output.albedo = getColorFromTexture(material.baseColorIndex, TEXTURE_BUFFERS_START, IN.uv, TextureUsage::Default, float4(0.f, 0.f, 0.f, 1.f));
	if (output.albedo.w < 1e-6f) {
		discard;
		return output; // discard; doesn't return so return here in order to save some texture reads.
	}

	output.albedo *= float4(material.baseColorFactor, 1.f);

	// inverse gamma-correction
	output.albedo.xyz = pow(output.albedo.xyz, 2.2);

	float3 normal = 0.f;
	if (sceneData.withNormalMapping && material.normalsIndex != INVALID_TEXTURE_INDEX) {
		float3 texNormal = getColorFromTexture(material.normalsIndex, TEXTURE_BUFFERS_START, IN.uv, TextureUsage::NormalMap).rgb;
		texNormal = texNormal * (255.f/127.f) - 128.f/127.f; // [0;1] -> [-1;1]

		// Flip y-coordinate. We are reading a normal map from glTF scene.
		// By glTF specification Y+ of the tangent space points up, whereas for DX12 Y+ points down
		// (right vs left handed systems).
		texNormal.y *= -1;
		
		texNormal = normalize(texNormal);
		
		normal = normalize(mul(IN.TBN, texNormal));
	} else {
		normal = IN.normal; // If we don't have a normal map or normal mapping is disabled just use the geometric normal
	}
	output.normal = float4(normal, 0.f);

	float4 metallicRoughness = getColorFromTexture(material.metallicRoughnessIndex, TEXTURE_BUFFERS_START, IN.uv);
	float4 occlusion = 1.f;
	// Occlusion may be encoded in the same texture as the metallicRoughness part, so make sure to skip the texture read in that case
	if (material.metallicRoughnessIndex == material.ambientOcclusionIndex) {
		occlusion = metallicRoughness;
	} else if (material.ambientOcclusionIndex != INVALID_TEXTURE_INDEX) {
		occlusion = getColorFromTexture(material.ambientOcclusionIndex, TEXTURE_BUFFERS_START, IN.uv);
	}

	// Read occlusion from the red channel as it may be encoded in the metallicRoughness texture
	// If it's in a separate texture is just gray so it doesn't matter from which channel we read.
	// glTF standart metalness/roughness/occlusion encoding
	// metalness - blue channel
	// roughness - green channel
	// occlusion - red channel, if present
	metallicRoughness.b *= material.metallicFactor;
	metallicRoughness.g *= material.roughnessFactor;
	output.metalnessRoughnessOcclusion = float4(metallicRoughness.b, metallicRoughness.g, occlusion.r, 0.f);

	return output;
}