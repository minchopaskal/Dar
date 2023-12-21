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

	viewMatrix = glm::rotate(viewMatrix, angle, axis);
}

void Camera::zoom(float zoomFactor) {
	switch (type) {
	case CameraType::Perspective:
		fov = std::min(std::max(30.f, fov / zoomFactor), 150.f);
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
	pitchAngle = std::min(std::max(-89.f, pitchAngle), 89.f);
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
	Vec3 forward = forwardVector;
	if (keepXZ) {
		auto pn = glm::dot(forward, Vec3UnitY()) * Vec3UnitY();
		forward = glm::normalize(forward - pn);
	}
	move(forward * amount);
}

void Camera::moveRight(float amount) {
	updateOrientation();
	Vec3 right = rightVector;
	if (keepXZ) {
		Vec3 forward = forwardVector;
		auto pn = glm::dot(forward, Vec3UnitY()) * Vec3UnitY();
		forward = glm::normalize(forward - pn);
		right = glm::normalize(glm::cross(Vec3UnitY(), forward));
	}
	move(rightVector * amount);
}

void Camera::moveUp(float amount) {
	updateOrientation();
	Vec3 up = keepXZ ? Vec3UnitY() : upVector;
	move(up * amount);
}

void Camera::updateOrientation() const {
	if (orientationValid) {
		return;
	}

	forwardVector.x = cos(glm::radians(pitchAngle)) * cos(glm::radians(yawAngle));
	forwardVector.y = sin(glm::radians(pitchAngle));
	forwardVector.z = sin(glm::radians(yawAngle)) * cos(glm::radians(pitchAngle));
	forwardVector = glm::normalize(forwardVector);
	rightVector = glm::normalize(glm::cross(Vec3UnitY(), forwardVector));
	upVector = glm::normalize(glm::cross(forwardVector, rightVector));

	orientationValid = true;
}

float Camera::getNearPlane() const {
	return nearPlane;
}

float Camera::getFarPlane() const {
	return farPlane;
}

void Camera::updateAspectRatio(unsigned int w, unsigned int h) {
	if (type == CameraType::Orthographic) {
		width = float(w);
		height = float(h);
	} else {
		aspectRatio = w / float(h);
	}

	projectionMatrixValid = false;
}

void Camera::updateViewMatrix() const {
	if (viewMatrixValid) {
		return;
	}

	updateOrientation();

	viewMatrix = glm::lookAt(pos, pos + forwardVector, Vec3UnitY());

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
		projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
		break;
	case CameraType::Orthographic:
		projectionMatrix = glm::ortho(-(width / 2), width / 2, -(height / 2), height / 2, nearPlane, farPlane);
		break;
	default:
		dassert(false);
		return Mat4(1.f);
	}

	projectionMatrixValid = true;
	return projectionMatrix;
}

} // namespace Dar