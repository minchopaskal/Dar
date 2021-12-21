#include "d3d12_camera.h"
#include "d3d12_defines.h"

Mat4 Camera::getViewMatrix() const {
	Mat3 rotMatrix = orientation.getRotationMatrix();

	Vec3 translation = -rotMatrix * pos;

	Mat4 result{ orientation.getRotationMatrix(), Vec4(0.f, 0.f, 0.f, 1.f) };
	result.row1.w = translation.x;
	result.row2.w = translation.y;
	result.row3.w = translation.z;

	return result;
}

void Camera::move(const Vec3 &magnitude) {
	pos = pos + magnitude;
}

void Camera::rotate(const Vec3 &axis, float angle) {
	orientation *= Quat::makeQuat(angle, axis);
}

void Camera::zoom(float zoomFactor) {
	switch (type) {
	case CameraType::Perspective:
		fov = dmath::min(dmath::max(30.f, atan(tan(fov) / zoomFactor)), 150.f);
		break;
	case CameraType::Orthographic:
		// TODO:
		break;
	default:
		dassert(false);
		break;
	}
}

Camera&& Camera::perspectiveCamera(const Vec3 &pos, float fov, float aspectRatio, float nearPlane, float farPlane) {
	Camera res;

	res.pos = pos;
	res.fov = fov;
	res.aspectRatio = aspectRatio;
	res.nearPlane = nearPlane;
	res.farPlane = farPlane;
	res.type = CameraType::Perspective;

	return std::move(res);
}

Camera&& Camera::orthographicCamera(const Vec3 &pos, float renderRectWidth, float renderRectHeight, float nearPlane, float farPlane) {
	Camera res;

	res.pos = pos;
	res.width = renderRectWidth;
	res.height = renderRectHeight;
	res.nearPlane = nearPlane;
	res.farPlane = farPlane;
	res.type = CameraType::Orthographic;

	return std::move(res);
}

Mat4 Camera::getProjectionMatrix() const {
	switch (type) {
	case CameraType::Perspective:
		return dmath::perspective(fov, aspectRatio, nearPlane, farPlane);
	case CameraType::Orthographic:
		return dmath::orthographic(-(width / 2), width / 2, -(height / 2), height / 2, nearPlane, farPlane);
	default:
		dassert(false);
		return Mat4(1.f);
	}
}
