#ifndef GPU_CPU_COMMON_HLSLI
#define GPU_CPU_COMMON_HLSLI

#include "interop.hlsli"

struct ShaderRenderData {
	Mat4 viewProjection;
	Mat4 invView;
	Mat4 invProjection;
	Vec4 cameraPosition; // world-space
	Vec4 cameraDir; // world-space
	float invWidth; // needed for FXAA
	float invHeight; // needed for FXAA
	float nearPlane;
	float farPlane;
	int numLights;
	int width;
	int height;
	UINT boneIdx;
	
	UINT frame;
	float delta;
	float time;

	// Options
	int showGBuffer;
	int withNormalMapping;
	int spotLightON;
	int fxaaON;
	int darken;
};

struct MeshData {
	Mat4 modelMatrix;
	Mat4 normalMatrix;
	UINT materialId;
};

struct MaterialData {
	Vec3 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	UINT baseColorIndex;
	UINT normalsIndex;
	UINT metallicRoughnessIndex;
	UINT ambientOcclusionIndex;
};

enum LightType {
	InvalidLight = -1,

	Point = 0,
	Directional,
	Spot,

	LightCount
};

static const int MAX_SHADOW_MAPS_COUNT = 2;

struct LightcasterDesc {
	int index;
};

struct LightData {
	Mat4 viewProjection;
	Vec3 position;
	Vec3 diffuse;
	Vec3 ambient;
	Vec3 specular;
	Vec3 attenuation;
	Vec3 direction;
	float innerAngleCutoff;
	float outerAngleCutoff;
	float zNear;
	float zFar;
	int type;
	int shadowMapIndexOffset;
};

enum TextureUsage {
	Default,
	NormalMap,
};

#endif // GPU_CPU_COMMON_HLSLI
