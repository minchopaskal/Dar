#ifndef GPU_CPU_COMMON_HLSLI
#define GPU_CPU_COMMON_HLSLI

#ifdef __HLSL_VERSION
typedef row_major matrix Mat4;
typedef float4 Vec4;
typedef float3 Vec3;
typedef uint UINT;
#endif // __HLSL_VERSION

struct SceneData {
	Mat4 viewProjection;
	Vec4 cameraPosition; // world-space
	Vec4 cameraDir; // world-space
	int numLights;
	int showGBuffer;
	int width;
	int height;
	int withNormalMapping;
	int spotLightOn;
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
	Invalid = -1,

	Point = 0,
	Directional,
	Spot,

	Count
};

struct LightData {
	Vec3 position;
	Vec3 diffuse;
	Vec3 ambient;
	Vec3 specular;
	Vec3 attenuation;
	Vec3 direction;
	float innerAngleCutoff;
	float outerAngleCutoff;
	int type;
};

#endif // GPU_CPU_COMMON_HLSLI