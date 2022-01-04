struct ModelViewProjection
{
	row_major matrix MVP;
};
 
ConstantBuffer<ModelViewProjection> MVPConstBuf : register(b0);

struct VSInput
{
	float3 position : POSITION;
	float2 uv : TEXCOORD;
};

struct VSOutput
{
	float4 position : SV_POSITION;
	float2 uv : TEXCOORD;
};

VSOutput main(VSInput IN)
{
	VSOutput result;

	result.position = mul(MVPConstBuf.MVP, float4(IN.position, 1.f));
	result.uv = IN.uv;
	
	return result;
}
