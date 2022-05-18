#include "framework/camera.h"
#include "utils/defines.h"

namespace Dar {

Mat4 Camera::getViewMatrix() const {
	if (type == CameraType::Invalid) {
		dassert(false);
		return Mat4(1.f);
	}

	updateViewMatrix();

	return viewMatrix;
}

void Camera::move(const Vec3 &magnitude) {
	pos = pos + magnitude;

	viewMatrixValid = false;
}

void Camera::rotate(const Vec3 &axis, float angle) {
	updateViewMatrix(); // update the view matrix if needed

	viewMatrix = viewMatrix.rotate(axis, angle);
}

void Camera::zoom(float zoomFactor) {
	switch (type) {
	case CameraType::Perspective:
		fov = dmath::min(dmath::max(30.f, fov / zoomFactor), 150.f);
		break;
	case CameraType::Orthographic:
		// TODO:
		break;
	default:
		dassert(false);
		break;
	}

	projectionMatrixValid = false;
}

void Camera::yaw(float angleDeg) {
	yawAngle -= angleDeg;
	orientationValid = false;
	viewMatrixValid = false;
}

void Camera::pitch(float angleDeg) {
	pitchAngle -= angleDeg;
	pitchAngle = dmath::min(dmath::max(-89.f, pitchAngle), 89.f);
	orientationValid = false;
	viewMatrixValid = false;
}

void Camera::roll(float angleDeg) {
	rollAngle -= angleDeg;
	orientationValid = false;
	viewMatrixValid = false;
}

void Camera::moveForward(float amount) {
	updateOrientation();
	float y = pos.y;
	move(forwardVector * amount);
	if (keepXZ) {
		pos.y = y;
	}
}

void Camera::moveRight(float amount) {
	updateOrientation();
	float y = pos.y;
	move(rightVector * amount);
	if (keepXZ) {
		pos.y = y;
	}
}

void Camera::moveUp(float amount) {
	updateOrientation();
	move(upVector * amount);
}

void Camera::updateOrientation() const {
	if (orientationValid) {
		return;
	}

	forwardVector.x = cos(dmath::radians(pitchAngle)) * cos(dmath::radians(yawAngle));
	forwardVector.y = sin(dmath::radians(pitchAngle));
	forwardVector.z = sin(dmath::radians(yawAngle)) * cos(dmath::radians(pitchAngle));
	forwardVector = forwardVector.normalized();
	rightVector = Vec3::unitY().cross(forwardVector).normalized();
	upVector = forwardVector.cross(rightVector).normalized();

	orientationValid = true;
}

void Camera::updateAspectRatio(unsigned int width, unsigned int height) {
	if (type == CameraType::Orthographic) {
		this->width = float(width);
		this->height = float(height);
	} else {
		aspectRatio = width / float(height);
	}

	projectionMatrixValid = false;
}

void Camera::updateViewMatrix() const {
	if (viewMatrixValid) {
		return;
	}

	updateOrientation();

	viewMatrix = dmath::lookAt(pos + forwardVector, pos, Vec3::unitY());

	viewMatrixValid = true;
}

Camera Camera::perspectiveCamera(const Vec3 &pos, float fov, float aspectRatio, float nearPlane, float farPlane) {
	Camera res;

	res.pos = pos;
	res.fov = fov;
	res.aspectRatio = aspectRatio;
	res.nearPlane = nearPlane;
	res.farPlane = farPlane;
	res.type = CameraType::Perspective;

	return res;
}

Camera Camera::orthographicCamera(const Vec3 &pos, float renderRectWidth, float renderRectHeight, float nearPlane, float farPlane) {
	Camera res;

	res.pos = pos;
	res.width = renderRectWidth;
	res.height = renderRectHeight;
	res.nearPlane = nearPlane;
	res.farPlane = farPlane;
	res.type = CameraType::Orthographic;

	return res;
}

Mat4 Camera::getProjectionMatrix() const {
	if (projectionMatrixValid) {
		return projectionMatrix;
	}

	switch (type) {
	case CameraType::Perspective:
		projectionMatrix = dmath::perspective(fov, aspectRatio, nearPlane, farPlane);
		break;
	case CameraType::Orthographic:
		projectionMatrix = dmath::orthographic(-(width / 2), width / 2, -(height / 2), height / 2, nearPlane, farPlane);
		break;
	default:
		dassert(false);
		return Mat4(1.f);
	}

	projectionMatrixValid = true;
	return projectionMatrix;
}

} // namespace Dar