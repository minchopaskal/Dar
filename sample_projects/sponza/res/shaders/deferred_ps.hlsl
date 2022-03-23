#include "common.hlsli"

struct PSInput
{
	float4 position : SV_Position;
	float4 fragPos : POSITION0;
	float3 normal : NORMAL;
	row_major float3x3 TBN : TANGENT_MATRIX; // tangent space -> world space
	float2 uv : TEXCOORD;
};

struct PSOutput
{
	float4 albedo;
	float4 normal;
	float4 metalnessRoughnessOcclusion;
	float4 position;
};

static const uint MATERIALS_BUFFER_INDEX = 0;
static const uint TEXTURE_BUFFERS_START = 1;

PSOutput main(PSInput IN) : SV_Target
{
	StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialData materialIndices = materials[meshData.materialId];

	PSOutput output;

	output.albedo = getColorFromTexture(materialIndices.baseColorIndex, TEXTURE_BUFFERS_START, IN.uv, float4(0.f, 0.f, 0.f, 1.f));
	if (output.albedo.w < 1e-6f) {
		discard;
		return output; // discard; doesn't return so return here in order to save some texture reads.
	}
	// inverse gamma-correction
	output.albedo.xyz = pow(output.albedo.xyz, 2.2);

	float3 normal = 0.f;
	if (sceneData.withNormalMapping && materialIndices.normalsIndex != uint(-1)) {
		float3 texNormal = getColorFromTexture(materialIndices.normalsIndex, TEXTURE_BUFFERS_START, IN.uv).rgb;
		texNormal = texNormal * (255.f/127.f) - 128.f/127.f;

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

	const float4 metallicRoughness = getColorFromTexture(materialIndices.metallicRoughnessIndex, TEXTURE_BUFFERS_START, IN.uv);
	float4 occlusion = 1.f;
	// Occlusion may be encoded in the same texture as the metallicRoughness part, so make sure to skip the texture read in that case
	if (materialIndices.metallicRoughnessIndex == materialIndices.ambientOcclusionIndex) {
		occlusion = metallicRoughness;
	} else {
		occlusion = getColorFromTexture(materialIndices.ambientOcclusionIndex, TEXTURE_BUFFERS_START, IN.uv);
	}

	// Read occlusion from the blue channel as it may be encoded in the metallicRoughness texture
	// If it's in a separate texture is just gray so it doesn't matter from which channel we read.
	output.metalnessRoughnessOcclusion = float4(metallicRoughness.r, metallicRoughness.g, occlusion.b, 0.f);

	output.position = IN.fragPos;

	return output;
}