#include "rasterizer_utils.cuh"

dvoid mat4::transpose() {
	mat4 c = *this;
	rows[0] = make_float4(c.rows[0].x, c.rows[1].x, c.rows[2].x, c.rows[3].x);
	rows[1] = make_float4(c.rows[0].y, c.rows[1].y, c.rows[2].y, c.rows[3].y);
	rows[2] = make_float4(c.rows[0].z, c.rows[1].z, c.rows[2].z, c.rows[3].z);
	rows[3] = make_float4(c.rows[0].w, c.rows[1].w, c.rows[2].w, c.rows[3].w);
}

__device__ mat4 operator*(mat4 m1, mat4 m2) {
	mat4 res;
	m2.transpose();
	res.rows[0] = make_float4(dot(m1.rows[0], m2.rows[0]), dot(m1.rows[0], m2.rows[1]), dot(m1.rows[0], m2.rows[2]), dot(m1.rows[0], m2.rows[3]));
	res.rows[1] = make_float4(dot(m1.rows[1], m2.rows[0]), dot(m1.rows[1], m2.rows[1]), dot(m1.rows[1], m2.rows[2]), dot(m1.rows[1], m2.rows[3]));
	res.rows[2] = make_float4(dot(m1.rows[2], m2.rows[0]), dot(m1.rows[2], m2.rows[1]), dot(m1.rows[2], m2.rows[2]), dot(m1.rows[2], m2.rows[3]));
	res.rows[3] = make_float4(dot(m1.rows[3], m2.rows[0]), dot(m1.rows[3], m2.rows[1]), dot(m1.rows[3], m2.rows[2]), dot(m1.rows[3], m2.rows[3]));

	return res;
}

dfloat4 operator*(mat4 m, float4 v) {
	return make_float4(
		dot(m.rows[0], v),
		dot(m.rows[1], v),
		dot(m.rows[2], v),
		dot(m.rows[3], v)
	);
}

dfloat3 operator*(mat4 m, float3 v) {
	float4 v4 = toFloat4(v);
	return make_float3(
		dot(m.rows[0], v4),
		dot(m.rows[1], v4),
		dot(m.rows[2], v4)
	);
}

dfloat4 sample(TextureSampler *sampler, float2 uv) {
	const int x = min(Max(0, uv.x * sampler->width), sampler->width - 1);
	const int y = min(Max(0, uv.y * sampler->height), sampler->height - 1);
	const int idx = (y * sampler->width + x) * sampler->numComp;

	return make_float4(
		(sampler->data[idx + 0] / 255.f),
		(sampler->data[idx + 1] / 255.f),
		(sampler->data[idx + 2] / 255.f),
		1.f
	);
}

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

dfloat4 operator/(float4 a, float b) {
	return make_float4(
		a.x / b,
		a.y / b,
		a.z / b,
		a.w / b
	);
}

__device__ float4& operator/=(float4 &a, float b) {
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;

	return a;
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
	return fmaf(v1.x, v2.x, fmaf(v1.y, v2.y, fmaf(v1.z, v2.z, 0.f)));
}

dfloat dot(float4 v1, float4 v2) {
	return fmaf(v1.x, v2.x, fmaf(v1.y, v2.y, fmaf(v1.z, v2.z, fmaf(v1.w, v2.w, 0.f))));
}

dfloat3 cross(const float3 v1, const float3 v2) {
	return make_float3(
		fmaf(v1.y, v2.z, -fmaf(v1.z, v2.y, 0.f)),
		fmaf(v1.z, v2.x, -fmaf(v1.x, v2.z, 0.f)),
		fmaf(v1.x, v2.y, -fmaf(v1.y, v2.x, 0.f))
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

dint Min(int a, int b) {
	return a < b ? a : b;
}

dint Max(int a, int b) {
	return a > b ? a : b;
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
