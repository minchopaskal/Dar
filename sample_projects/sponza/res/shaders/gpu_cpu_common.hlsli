#ifdef __HLSL_VERSION
typedef row_major matrix Mat4;
typedef float4 Vec4;
typedef uint UINT;
#endif // DXC_VERSION_MAJOR

struct SceneData {
	Mat4 viewProjection;
	Vec4 cameraPosition; // world-space
	Vec4 cameraDir; // world-space
	int numLights;
	int showGBuffer;
	int width;
	int height;
	int withNormalMapping;
};

struct MeshData {
	Mat4 modelMatrix;
	Mat4 normalMatrix;
	UINT materialId;
};
