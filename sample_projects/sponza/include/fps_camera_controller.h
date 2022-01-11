#pragma once

#include "d3d12_camera.h"

struct Camera;

struct FPSCameraController : public ICameraController {
	FPSCameraController(Camera *cam, double movementSpeed);

	double getSpeed() const { return speed; }

	// Inherited via CameraController
	void onMouseMove(double xPos, double yPos, double deltaTime) override;

	void onMouseScroll(double xOffset, double yOffset, double deltaTime) override;

	void processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltaTime) override;

	virtual void onDrawUI() override;

private:
	struct MousePos {
		double x;
		double y;
	} mousePos;

	double speed;
	double mouseSensitivity;
	bool mousePosValid;
};
