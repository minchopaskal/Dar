#ifndef GPU_CPU_COMMON_HLSLI
#define GPU_CPU_COMMON_HLSLI

#ifdef __HLSL_VERSION
typedef row_major matrix Mat4;
typedef float4 Vec4;
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

#ifdef __HLSL_VERSION
struct MaterialData {
	float3 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	UINT baseColorIndex;
	UINT normalsIndex;
	UINT metallicRoughnessIndex;
	UINT ambientOcclusionIndex;
};
#endif // __HLSL_VERSION

#endif // GPU_CPU_COMMON_HLSLI