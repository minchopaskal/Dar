struct VSOutput {
	float2 uv : TEXCOORD;
	float4 pos : SV_Position;
};

VSOutput main(in uint vertID : SV_VertexID) {
	VSOutput result;

	result.uv = float2(
		(float)(vertID / 2) * 2.f ,
		1.f - (float)(vertID % 2) * 2.f
	);

	result.pos = float4(
		(float)(vertID / 2) * 4.f - 1.f,
		(float)(vertID % 2) * 4.f - 1.f,
		0.f,
		1.f
	);

	return result;
}
