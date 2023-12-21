#ifndef INTEROP_HLSLI
#define INTEROP_HLSLI

#ifdef __HLSL_VERSION
typedef matrix Mat4;
typedef float4 Vec4;
typedef float3 Vec3;
typedef uint UINT;
#endif // __HLSL_VERSION

#endif // INTEROP_HLSLI
