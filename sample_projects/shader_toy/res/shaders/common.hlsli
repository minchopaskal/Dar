struct ConstData {
	int width;
	int height;
	int frame;
	float delta;
	int hasOutput;
};

ConstantBuffer<ConstData> constData : register(b0);
SamplerState Sampler : register(s0);

struct PSInput {
	float2 uv : TEXCOORD;
};

