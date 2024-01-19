#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include "glm/glm.hpp"
#include "glm/ext.hpp"

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Mat4x3 = glm::mat4x3;
using Quaternion = glm::quat;

static inline Vec3 Vec3UnitX() {
	return Vec3(1.f, 0.f, 0.f);
}

static inline Vec3 Vec3UnitY() {
	return Vec3(0.f, 1.f, 0.f);
}

static inline Vec3 Vec3UnitZ() {
	return Vec3(0.f, 0.f, 1.f);
}
