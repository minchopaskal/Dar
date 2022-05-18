#pragma once

#include "framework/camera.h"

struct GLFWwindow;

struct FPSCameraController : public Dar::ICameraController {
	FPSCameraController(Dar::Camera *cam, double movementSpeed);

	double getSpeed() const { return speed; }

	const Dar::Camera& getCamera() const {
		return *cam;
	}

	// Inherited via CameraController
	virtual void onMouseMove(double xPos, double yPos, double deltaTime) override;

	virtual void onMouseScroll(double xOffset, double yOffset, double deltaTime) override;

	virtual void processKeyboardInput(Dar::IKeyboardInputQuery *inputQuery, double deltaTime) override;

public:
	GLFWwindow *window;

protected:
	struct MousePos {
		double x;
		double y;
	} mousePos;

	double speed;
	double mouseSensitivity;
	bool mousePosValid;
	bool shiftPressed = false;
	bool altPressed = false;
};
