#pragma once

#include "d3d12_math.h"

enum class CameraType : int {
	Invalid = 0,
	Perspective,
	Orthographic,

	Count
};

// TODO: implement cameras import
struct Camera {
	static Camera&& perspectiveCamera(const Vec3 &pos, float fov, float aspectRatio, float nearPlane, float farPlane);
	static Camera&& orthographicCamera(const Vec3 &pos, float renderRectWidth, float renderRectHeight, float nearPlane, float farPlane);

	Camera(Camera&&) = default;
	Camera& operator=(Camera&&) = default;

	/// Get the world-to-camera transformation
	Mat4 getViewMatrix() const;

	/// Get the camera-to-clip transformation
	Mat4 getProjectionMatrix() const;

	/// Add `magnitude` to the camera position
	void move(const Vec3 &magnitude);

	/// Rotate the camera around `axis` by `angle` degrees
	void rotate(const Vec3 &axis, float angle);

	/// Zoom by factor
	void zoom(float factor);

private:
	Camera() : fov(90.f), aspectRatio(1.f) { }

private:
	Quat orientation = Quat::makeQuat(0.f, Vec3(0.f, 1.f, 0.f));
	Vec3 pos = Vec3(0.f, 0.f, 0.f);

	// Frustum
	// TODO: probably abstraction for the frustum?
	union {
		struct /*PerspectiveData*/ {
			float fov; ///< vertical FOV
			float aspectRatio; ///< aspect ratio
		};

		struct /*OrthographicData*/ {
			float width;
			float height;
		};
	};

	float nearPlane = 0.001f;
	float farPlane = 100.f;

	CameraType type = CameraType::Invalid;
};