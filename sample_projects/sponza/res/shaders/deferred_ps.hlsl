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
	float4 diffuse;
	float4 specular;
	float4 normal;
	float4 position;
};

static const uint MATERIALS_BUFFER_INDEX = 0;
static const uint TEXTURE_BUFFERS_START = 1;

PSOutput main(PSInput IN) : SV_Target
{
	StructuredBuffer<MaterialIndices> materials = ResourceDescriptorHeap[MATERIALS_BUFFER_INDEX];
	const MaterialIndices materialIndices = materials[meshData.materialId];

	PSOutput output;

	output.diffuse = getColorFromTexture(materialIndices.diffuseIndex + TEXTURE_BUFFERS_START, IN.uv, float4(0.f, 0.f, 0.f, 1.f));
	if (output.diffuse.w < 1e-6f) {
		discard;
		return output; // discard; doesn't return so return here in order to save some texture reads.
	}

	// inverse gamma-correction
	output.diffuse.xyz = pow(output.diffuse.xyz, 2.2);

	output.specular = getColorFromTexture(materialIndices.specularIndex + TEXTURE_BUFFERS_START, IN.uv, float4(.5f, .5f, .5f, 1.f));

	output.position = IN.fragPos;

	float3 normal = 0.f;
	if (sceneData.withNormalMapping && materialIndices.normalIndex != uint(-1)) {
		float3 texNormal = getColorFromTexture(materialIndices.normalIndex + TEXTURE_BUFFERS_START, IN.uv).rgb;
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

	return output;
}