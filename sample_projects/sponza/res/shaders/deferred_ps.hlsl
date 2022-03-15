#include "common.hlsli"

struct PSInput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
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
	// TODO: normal mapping.
	output.normal = float4(IN.normal, 0.f);//getColorFromTexture(materialIndices.normalIndex + TEXTURE_BUFFERS_START, IN.uv, float4(IN.normal, 0.f));
	output.normal = float4(normalize(output.normal.xyz), 0.f);
	output.position = IN.fragPos;

	return output;
}