struct ConstData {
	int width;
	int height;
	uint frame;
	float delta;
	int hasOutput;
	float2 seed;
};

ConstantBuffer<ConstData> constData : register(b0);
SamplerState Sampler : register(s0);

struct PSInput {
	float2 uv : TEXCOORD;
};

Texture2D<float4> getPreviousFrameTexture(int renderTargetIndex) {
	Texture2D<float4> res = ResourceDescriptorHeap[renderTargetIndex];
	return res;
}

// TODO: get num render targets from const data or smth
Texture2D<float4> getTextureResource(int numRenderTargets, int textureResourceIndex) {
	Texture2D<float4> res = ResourceDescriptorHeap[textureResourceIndex + numRenderTargets];
	return res;
}

// Some pseudo-random generators taken from stackoverflow
float randFloat(float2 uv) {
	float2 noise = (frac(sin(dot(uv, float2(12.9898, 78.233) * 2.0)) * 43758.5453));
	return abs(noise.x + noise.y) * 0.5;
}

float2 randFloat2(float2 uv) {
	float noiseX = (frac(sin(dot(uv, float2(12.9898, 78.233) * 2.0)) * 43758.5453));
	float noiseY = sqrt(1 - noiseX * noiseX);
	return float2(noiseX, noiseY);
}

float2 randFloat2_2(in float2 uv) {
	float noiseX = (frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453));
	float noiseY = (frac(sin(dot(uv, float2(12.9898, 78.233) * 2.0)) * 43758.5453));
	return float2(noiseX, noiseY) * 0.004;
}