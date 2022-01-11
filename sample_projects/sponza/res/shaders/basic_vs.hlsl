struct ModelViewProjection
{
	row_major matrix normalMatrix;
	row_major matrix model;
	row_major matrix viewProjection;
};
 
ConstantBuffer<ModelViewProjection> MVPConstBuf : register(b0);

struct VSInput
{
	float3 position : POSITION0;
	float3 normal: NORMAL;
	float2 uv : TEXCOORD;
};

struct VSOutput
{
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
	float4 fragPos : POSITION0;
	float4 position : SV_Position;
};

VSOutput main(VSInput IN)
{
	VSOutput result;

	result.fragPos = mul(MVPConstBuf.model, float4(IN.position, 1.f));
	result.position = mul(MVPConstBuf.viewProjection, result.fragPos);
	result.normal = mul(MVPConstBuf.normalMatrix, float4(IN.normal, 1.f)).xyz; // model is orthogonal matrix
	result.uv = IN.uv;
	
	return result;
}
