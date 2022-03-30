#include "common.hlsli"

struct PSInput {
	float2 uv : TEXCOORD;
};

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

static float contrastThreshold = 0.0312f;
static float relativeThreshold = 0.063f;

struct FXAAData {
	float3 colors[pixelDirection_Count];
};

struct FXAALumaData {
	float c[pixelDirection_Count];
	float contrast;
};

struct EdgeData {
	float blendStep;
	bool isHorizontal;
};

EdgeData determineEdge(FXAALumaData l) {
	EdgeData e;
	float horizontal =
		abs(l.c[N] + l.c[S] - 2 * l.c[M]) * 2 +
		abs(l.c[NE] + l.c[SE] - 2 * l.c[E]) +
		abs(l.c[NW] + l.c[SW] - 2 * l.c[W]);
	float vertical =
		abs(l.c[E] + l.c[W] - 2 * l.c[M]) * 2 +
		abs(l.c[NE] + l.c[NW] - 2 * l.c[N]) +
		abs(l.c[SE] + l.c[SW] - 2 * l.c[S]);
	e.isHorizontal = horizontal >= vertical;

	float pLuma = e.isHorizontal ? l.c[S] : l.c[E];
	float nLuma = e.isHorizontal ? l.c[N] : l.c[W];
	float pGradient = abs(pLuma - l.c[M]);
	float nGradient = abs(nLuma - l.c[M]);

	e.blendStep = e.isHorizontal ? sceneData.invWidth : sceneData.invHeight;
	if (pGradient < nGradient) {
		e.blendStep = -e.blendStep;
	}

	return e;
}

float luminance(float3 color) {
	return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float calcPixelBlendFactor(FXAALumaData luma) {
	float avg = 2 * (luma.c[N] + luma.c[S] + luma.c[W] + luma.c[E]);
	avg += (luma.c[NW] + luma.c[NE] + luma.c[SW] + luma.c[SE]);
	avg *= 1 / 12.f;
	
	float filter = abs(luma.c[M] - avg);

	filter = clamp(filter / luma.contrast, 0.f, 1.f);

	float blendFactor = smoothstep(0.f, 1.f, filter);

	return blendFactor * blendFactor;
}

float4 fxaaFilter(FXAAData data, Texture2D<float4> renderTex, float2 uv) {
	FXAALumaData luma;
	luma.c[M] = luminance(data.colors[M]);
	luma.c[N] = luminance(data.colors[N]);
	luma.c[E] = luminance(data.colors[E]);
	luma.c[S] = luminance(data.colors[S]);
	luma.c[W] = luminance(data.colors[W]);
	luma.c[NW] = luminance(data.colors[NW]);
	luma.c[NE] = luminance(data.colors[NE]);
	luma.c[SW] = luminance(data.colors[SW]);
	luma.c[SE] = luminance(data.colors[SE]);

	float lumaMin = min(luma.c[M], min(luma.c[N], min(luma.c[E], min(luma.c[S], luma.c[W]))));
	float lumaMax = max(luma.c[M], max(luma.c[N], max(luma.c[E], max(luma.c[S], luma.c[W]))));
	luma.contrast = lumaMax - lumaMin;

	if (luma.contrast < relativeThreshold * lumaMax || luma.contrast < contrastThreshold) {
		return float4(data.colors[M], 1.f);
	}

	float blendFactor = calcPixelBlendFactor(luma);

	EdgeData e = determineEdge(luma);

	float blendValue = e.blendStep * blendFactor;
	if (e.isHorizontal) {
		uv.x += blendValue;
	} else {
		uv.y += blendValue;
	}

	float4 c = renderTex.SampleLevel(Sampler, uv, 0);
	c = c / (c + 1.f);
	c = pow(c, 1 / 2.2);
	return c;
}

float4 main(PSInput IN) : SV_TARGET{
	Texture2D<float4> renderTex = ResourceDescriptorHeap[0];
	Texture2D<float> depthTex = ResourceDescriptorHeap[1];

	int3 p = int3(IN.uv.x * sceneData.width, IN.uv.y * sceneData.height, 0);
	float4 color = renderTex.Load(p);

	FXAAData fxaaData;
	for (int i = 0; i < pixelDirection_Count; ++i) {
		int3 index = p + pixelOffset[i];
		if (index.x < 0 || index.x >= sceneData.width || index.y < 0 || index.y >= sceneData.height) {
			fxaaData.colors[i] = 0.f; // ?
		} else {
			fxaaData.colors[i] = renderTex.Load(index).rgb;
		}

		// TODO: this should be in a separate pass so we don't do
		// that multiple times per pixel.
		
		// tone-mapping
		fxaaData.colors[i] = fxaaData.colors[i] / (fxaaData.colors[i] + 1.f);

		// gamma-correction
		fxaaData.colors[i] = pow(fxaaData.colors[i], 1 / 2.2);
	}

	// tone-mapping
	color = color / (color + 1.f);

	// apply gamma-correction
	color = pow(color, 1/2.2);

	if (sceneData.fxaaON) {
		return fxaaFilter(fxaaData, renderTex, IN.uv);
	}

	return color;
}