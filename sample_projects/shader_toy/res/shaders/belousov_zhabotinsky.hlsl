#include "common.hlsli"

// Base on the paper
// https://discovery.ucl.ac.uk/id/eprint/17241/1/17241.pdf

static float alpha = 1.2f;
static float beta = 1.f;
static float gamma = 1.f;

// We don't want to update each frame as it will
// just die out really quickly. If the reaction is
// too slow for you, lower this paramer,
// if it's too fast - make it bigger.
static float updateConstant = 5.f;

// Initial pixel size
static float pixelSize = 256.f;

enum PixelDirection {
	NW = 0,
	N,
	NE,
	W,
	M,
	E,
	SW,
	S,
	SE,

	pixelDirection_Count
};

static int3 pixelOffset[pixelDirection_Count] = {
	int3(-1, -1, 0), int3(0, -1, 0), int3(1, -1, 0),
	int3(-1, 0, 0), int3(0, 0, 0), int3(1, 0, 0),
	int3(-1, 1, 0), int3(0, 1, 0), int3(1, 1, 0)
};

float4 main(PSInput IN) : SV_TARGET
{
	const float2 pixSize = float2( pixelSize, pixelSize * constData.height / constData.width);
	float2 uv = round(IN.uv * pixSize) / pixSize;
	// Generate bigger pixels at the start so we have more interesting reactions.
	if (constData.frame < 1) {
		return float4(randFloat(constData.seed * uv), randFloat2(constData.seed * uv), 1.f);
	}

	Texture2D<float4> p = getPreviousFrameTexture(0);
	const int x = constData.width * IN.uv.x;
	const int y = constData.height * IN.uv.y;
	const int3 index = int3(x, y, 0);

	if (constData.frame % updateConstant != 0) {
		return p.Load(index);
	}
	
	// Average all neighbouring pixels
	float4 pc = 0.f;
	for (int i = 0; i < pixelDirection_Count; ++i) {
		int3 currentIndex = index;
		currentIndex.x = (index.x + pixelOffset[i].x + constData.width) % constData.width;
		currentIndex.y = (index.y + pixelOffset[i].y + constData.height) % constData.height;

		pc += p.Load(currentIndex);
	}
	pc /= pixelDirection_Count;

	// React!
	float4 result = 1.f;
	result.x = pc.x + pc.x * (alpha * pc.y - gamma * pc.z);
	result.y = pc.y + pc.y * (beta * pc.z - alpha * pc.x);
	result.z = pc.z + pc.z * (gamma * pc.x - beta * pc.y);
	
	return clamp(result, 0.f, 1.f);
}





