struct ConstData {
	int width;
	int height;
	uint frame;
	float delta;
	int hasOutput;
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
