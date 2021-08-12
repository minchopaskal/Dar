#ifndef RASTERIZER_UTILS_CU
#define RASTERIZER_UTILS_CU

#define gvoid   __global__ void
#define gint    __global__ int
#define gint2   __global__ int2
#define gint3   __global__ int3
#define gint4   __global__ int4
#define guint   __global__ unsigned int
#define guchar  __global__ unsigned char
#define gbool   __global__ bool
#define gfloat  __global__ float
#define gfloat2 __global__ float2
#define gfloat3 __global__ float3
#define gfloat4 __global__ float4

#define dvoid   __device__ void
#define dint    __device__ int
#define dint2   __device__ int2
#define dint3   __device__ int3
#define dint4   __device__ int4
#define duint   __device__ unsigned int
#define duchar  __device__ unsigned char
#define dbool   __device__ bool
#define dfloat  __device__ float
#define dfloat2 __device__ float2
#define dfloat3 __device__ float3
#define dfloat4 __device__ float4

#define cvoid   __constant__ void
#define cint    __constant__ int
#define cint2   __constant__ int2
#define cint3   __constant__ int3
#define cint4   __constant__ int4
#define cuint   __constant__ unsigned int
#define cuchar  __constant__ unsigned char
#define cbool   __constant__ bool
#define cfloat  __constant__ float
#define cfloat2 __constant__ float2
#define cfloat3 __constant__ float3
#define cfloat4 __constant__ float4

#define FORCEINLINE __forceinline__

FORCEINLINE dfloat3 fromFloat4(float4 v) {
	return make_float3(v.x, v.y, v.z);
}

FORCEINLINE dfloat4 toFloat4(float3 v) {
	return make_float4(v.x, v.y, v.z, 1.f);
}

FORCEINLINE dfloat4 operator+(float4 a, float4 b) {
	return make_float4(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

FORCEINLINE dfloat4 operator-(float4 a, float4 b) {
	return make_float4(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

FORCEINLINE dfloat4 operator*(float4 a, float b) {
	return make_float4(
		a.x * b,
		a.y * b,
		a.z * b,
		a.w * b
		);
}

FORCEINLINE dfloat4 operator*(float b, float4 a) {
	return make_float4(
		a.x * b,
		a.y * b,
		a.z * b,
		a.w * b
		);
}

FORCEINLINE dfloat dot(float3 v1, float3 v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

FORCEINLINE dfloat dot(float4 v1, float4 v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

FORCEINLINE dfloat3 cross(const float3 v1, const float3 v2) {
	return make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x
	);
}

FORCEINLINE dfloat3 normalize(float3 v) {
	float length = sqrt(dot(v, v));
	return make_float3(
		v.x / length,
		v.y / length,
		v.z / length
	);
}

FORCEINLINE dfloat4 normalizef4(float4 v) {
	float length = sqrt(dot(v, v));
	return make_float4(
		v.x / length,
		v.y / length,
		v.z / length,
		v.w / length
	);
}

extern "C" {
	// Needed for CudaManager::testSystem();
	cint arrSize;
	gvoid adder(int *arrA, int *arrB, int *result) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, arrSize - 1);
		result[idx] = arrA[idx] + arrB[idx];
	}
} // extern "C"

#endif // RASTERIZER_UTILS_CU
