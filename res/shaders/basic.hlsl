struct Input {
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

Input VSMain(float4 position : POSITION, float4 color : COLOR) {
	Input result;

	result.position = position;
	result.color = color;

	return result;
}

float4 PSMain(Input input) : SV_TARGET
{
	return input.color;
}