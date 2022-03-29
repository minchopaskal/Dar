#pragma once

#include "math/dar_math.h"

enum class CameraType : int {
	Invalid = 0,
	Perspective,
	Orthographic,

	Count
};

struct Camera {
	static Camera perspectiveCamera(const Vec3 &pos, float fov, float aspectRatio, float nearPlane, float farPlane);
	static Camera orthographicCamera(const Vec3 &pos, float renderRectWidth, float renderRectHeight, float nearPlane, float farPlane);

	Camera() : fov(90.f), aspectRatio(1.f) { }
	Camera(Vec3 forward, Vec3 up, Vec3 right) : 
		fov(90.f),
		aspectRatio(1.f),
		forwardVector(forward), 
		upVector(up),
		rightVector(right)
	{ }

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

	void yaw(float angleDeg);
	void pitch(float angleDeg);
	void roll(float angleDeg);

	/// Move camera along its forward vector.
	/// Negative amount would move the camera backwards.
	/// @param amount scaling factor for the unit forward vector.
	void moveForward(float amount);

	/// Move camera along its sideways vector.
	/// Negative amount would move the camera in the opposite direction.
	/// @param amount scaling factor for the unit sideways vector.
	void moveRight(float amount);

	/// Same as forward() and right()
	void moveUp(float amount);

	void setKeepXZPlane(bool keep) { keepXZ = keep; }
	bool getKeepXZPlane() const { return keepXZ; }

	// getCameraX/Y/Z() returns the coordinate system of the camera in world-space coordinates
	Vec3 getCameraZ() const {
		updateOrientation();
		return forwardVector;
	}

	Vec3 getCameraX() const {
		updateOrientation();
		return rightVector;
	}

	Vec3 getCameraY() const {
		updateOrientation();
		return upVector;
	}

	Vec3 getPos() const {
		return pos;
	}

	float getFOV() const {
		return fov;
	}

	void updateAspectRatio(unsigned int width, unsigned int height);

private:
	void updateOrientation() const;
	void updateViewMatrix() const;

private:
	mutable Mat4 viewMatrix = Mat4(1.f);
	mutable Mat4 projectionMatrix = Mat4(1.f);

	Vec3 pos = Vec3(0.f, 0.f, 0.f);
	
	mutable Vec3 forwardVector = Vec3::unitZ();
	mutable Vec3 upVector = Vec3::unitY();
	mutable Vec3 rightVector = Vec3::unitX();

	CameraType type = CameraType::Invalid;

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

	float yawAngle = 0.f;
	float pitchAngle = 0.f;
	float rollAngle = 0.f;

	bool keepXZ = false;
	mutable bool orientationValid = false;
	mutable bool viewMatrixValid = false;
	mutable bool projectionMatrixValid = false;
};

struct IKeyboardInputQuery;

struct ICameraController {
	ICameraController(Camera *cam) : cam(cam) { }

	void setCamera(Camera *cam) {
		this->cam = cam;
	}

	virtual void onMouseMove(double xPos, double yPos, double deltaTime) = 0;
	virtual void onMouseScroll(double xOffset, double yOffset, double deltaTime) = 0;
	virtual void processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltatTime) = 0;

protected:
	Camera *cam;
};
