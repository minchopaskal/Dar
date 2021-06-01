struct VSInput
{
	float4 position : POSITION;
	float4 color : COLOR;
};

struct VSOutput
{
	float4 color : COLOR;
	float4 position : SV_POSITION;
};

VSOutput main(VSInput IN)
{
	VSOutput result;

	result.color = IN.color;
	result.position = IN.position;
	
	return result;
}
