#pragma once

#include "framework/camera.h"

struct Dar::Camera;

struct FPSCameraController : public Dar::ICameraController {
	FPSCameraController(Dar::Camera *cam, double movementSpeed);

	double getSpeed() const { return speed; }

	// Inherited via CameraController
	void onMouseMove(double xPos, double yPos, double deltaTime) override;

	void onMouseScroll(double xOffset, double yOffset, double deltaTime) override;

	void processKeyboardInput(Dar::IKeyboardInputQuery *inputQuery, double deltaTime) override;

private:
	struct MousePos {
		double x;
		double y;
	} mousePos;

	double speed;
	double mouseSensitivity;
	bool mousePosValid;
};
