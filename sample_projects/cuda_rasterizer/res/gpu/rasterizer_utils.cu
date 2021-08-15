#include "rasterizer_utils.cuh"

dfloat3 fromFloat4(float4 v) {
	return make_float3(v.x, v.y, v.z);
}

dfloat4 toFloat4(float3 v) {
	return make_float4(v.x, v.y, v.z, 1.f);
}

dfloat2 operator+(float2 a, float2 b) {
	return make_float2(
		a.x + b.x,
		a.y + b.y
	);
}

dfloat3 operator+(float3 a, float3 b) {
	return make_float3(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	);
}

dfloat4 operator+(float4 a, float4 b) {
	return make_float4(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
		a.w + b.w
	);
}

dint2 operator-(int2 a, int2 b) {
	return make_int2(
		a.x - b.x,
		a.y - b.y
	);
}

dfloat2 operator-(float2 a, float2 b) {
	return make_float2(
		a.x - b.x,
		a.y - b.y
	);
}

dfloat4 operator-(float4 a, float4 b) {
	return make_float4(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
		a.w - b.w
	);
}

dfloat4 operator*(float4 a, float b) {
	return make_float4(
		a.x * b,
		a.y * b,
		a.z * b,
		a.w * b
		);
}

dfloat4 operator*(float b, float4 a) {
	return make_float4(
		a.x * b,
		a.y * b,
		a.z * b,
		a.w * b
		);
}

dfloat2 operator*(float b, float2 a) {
	return make_float2(
		a.x * b,
		a.y * b
	);
}

dfloat2 operator*(float2 a, float b) {
	return make_float2(
		a.x * b,
		a.y * b
	);
}


dfloat3 operator*(float3 a, float b) {
	return make_float3(
		a.x * b,
		a.y * b,
		a.z * b
	);
}

dfloat3 operator*(float b, float3 a) {
	return make_float3(
		a.x * b,
		a.y * b,
		a.z * b
	);
}

dfloat dot(float3 v1, float3 v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

dfloat dot(float4 v1, float4 v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

dfloat3 cross(const float3 v1, const float3 v2) {
	return make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x
	);
}

dfloat3 normalize(float3 v) {
	float length = sqrt(dot(v, v));
	return make_float3(
		v.x / length,
		v.y / length,
		v.z / length
	);
}

dfloat4 normalizef4(float4 v) {
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
