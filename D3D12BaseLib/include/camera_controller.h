#pragma once

#include "d3d12_input_query.h"

struct Camera;

struct CameraController {
	CameraController(Camera *cam) : cam(cam) { }

	virtual void onMouseMove(double xPos, double yPos, double deltaTime) = 0;
	virtual void onMouseScroll(double xOffset, double yOffset, double deltaTime) = 0;
	virtual void processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltatTime) = 0;

protected:
	Camera *cam;
};

struct FPSCameraController : public CameraController {
	FPSCameraController(Camera *cam, double movementSpeed);

	// Inherited via CameraController
	void onMouseMove(double xPos, double yPos, double deltaTime) override;

	void onMouseScroll(double xOffset, double yOffset, double deltaTime) override;

	void processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltaTime) override;

	double getSpeed() const { return speed; }

private:
	struct MousePos {
		double x;
		double y;
	} mousePos;

	double speed;
	double mouseSensitivity;
	bool mousePosValid;
};
