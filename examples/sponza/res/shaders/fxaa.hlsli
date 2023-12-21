#ifndef FXAA_HLSLI
#define FXAA_HLSLI

#include "gpu_cpu_common.hlsli"

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
static int edgeWalkSteps = 10;

struct FXAAData {
	float3 colors[pixelDirection_Count];
};

struct FXAALumaData {
	float c[pixelDirection_Count];
	float contrast;
};

struct EdgeData {
	float blendStep;
	float oppositeLuma;
	float gradient;
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
		e.oppositeLuma = pLuma;
		e.gradient = nGradient;
	} else {
		e.oppositeLuma = nLuma;
		e.gradient = pGradient;
	}

	return e;
}

float luminance(float3 color) {
	return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float sampleLuminance(Texture2D<float4> renderTex, float2 uv) {
	return luminance(renderTex.SampleLevel(Sampler, uv, 0).rgb);
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

float calcEdgeBlendFactor(Texture2D<float4> renderTex, EdgeData e, float luma, float2 uv) {
	float2 uvEdge = uv;
	float2 edgeStep = 0.f;
	if (e.isHorizontal) {
		uvEdge.y += e.blendStep * 0.5f;
		edgeStep = float2(sceneData.invWidth, 0.f);
	} else {
		uvEdge.x += e.blendStep * 0.5f;
		edgeStep = float2(0.f, sceneData.invHeight);
	}

	float edgeLuma = (luma + e.oppositeLuma) * 0.5f; // luminance of the exact point of the edge - average of pixel luminance and luminance of opposite (relating to the edge) pixel
	float gradientThreshold = e.gradient * 0.25f; // 1/4 of the gradient of the edge as a threshold

	float2 puv = uvEdge;
	float2 nuv = uvEdge;
	float pLumaDelta = 0.f;
	float nLumaDelta = 0.f;
	bool endOfEdgeP = false;
	bool endOfEdgeN = false;

	for (int i = 0; i < edgeWalkSteps; ++i) {
		if (!endOfEdgeP) { // Walk in the positive direction
			puv += edgeStep;
			pLumaDelta = sampleLuminance(renderTex, puv) - edgeLuma;
			endOfEdgeP = abs(pLumaDelta) >= gradientThreshold;
		} else if (i == edgeWalkSteps - 1) {
			puv += edgeStep;
		}

		if (!endOfEdgeN) { // Walk in the negative direction
			nuv -= edgeStep;
			nLumaDelta = sampleLuminance(renderTex, nuv) - edgeLuma;
			endOfEdgeN = abs(nLumaDelta) >= gradientThreshold;
		} else if (i == edgeWalkSteps - 1) {
			nuv += edgeStep;
		}
	}

	float pDist = abs(e.isHorizontal ? puv.x - uv.x : puv.y - uv.y);
	float nDist = abs(e.isHorizontal ? nuv.x - uv.x : nuv.y - uv.y);
	float shortestDist = min(pDist, nDist);
	bool deltaSign = (pDist < nDist ? (pLumaDelta >= 0) : (nLumaDelta >= 0));

	if (deltaSign == (luma - edgeLuma >= 0)) {
		return 0.f;
	}

	return 0.5f - shortestDist / (pDist + nDist);
}

FXAAData getFXAAData(Texture2D<float4> renderTex, int3 p) {
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

	return fxaaData;
}

float4 fxaaFilter(Texture2D<float4> renderTex, float2 uv) {
	int3 p = int3(uv.x * sceneData.width, uv.y * sceneData.height, 0);
	FXAAData data = getFXAAData(renderTex, p);

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

	EdgeData e = determineEdge(luma);

	float edgeBlendFactor = calcEdgeBlendFactor(renderTex, e, luma.c[M], uv);
	float pixelBlendFactor = calcPixelBlendFactor(luma);

	float blendFactor = max(edgeBlendFactor, pixelBlendFactor);

	float blendValue = e.blendStep * blendFactor;
	if (e.isHorizontal) {
		uv.x += blendValue;
	} else {
		uv.y += blendValue;
	}
	
	float4 c = renderTex.Sample(Sampler, uv);
	c.xyz = c.xyz / (c.xyz + 1.f);
	c.xyz = pow(c.xyz, 1/2.2f);
	return c;
}

#endif // FXAA_HLSLI
