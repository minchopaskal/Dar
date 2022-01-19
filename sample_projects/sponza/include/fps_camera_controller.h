#pragma once

#include "d3d12_camera.h"

struct Camera;

struct FPSCameraController : public ICameraController {
	FPSCameraController(Camera *cam, double movementSpeed);

	double getSpeed() const { return speed; }

	// Inherited via CameraController
	virtual void onMouseMove(double xPos, double yPos, double deltaTime) override;

	virtual void onMouseScroll(double xOffset, double yOffset, double deltaTime) override;

	virtual void processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltaTime) override;

	virtual void onDrawUI() override;

protected:
	struct MousePos {
		double x;
		double y;
	} mousePos;

	double speed;
	double mouseSensitivity;
	bool mousePosValid;
};
