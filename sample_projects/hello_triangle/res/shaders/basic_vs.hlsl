struct ModelViewProjection
{
	matrix MVP;
};
 
ConstantBuffer<ModelViewProjection> MVPConstBuf : register(b0);

struct VSInput
{
	float4 position : POSITION;
	float4 color : COLOR;
};

struct VSOutput
{
	float4 color : COLOR;
	float4 position : SV_Position;
};

VSOutput main(VSInput IN)
{
	VSOutput result;

	result.color = IN.color;
	result.position = mul(MVPConstBuf.MVP, IN.position);
	
	return result;
}
