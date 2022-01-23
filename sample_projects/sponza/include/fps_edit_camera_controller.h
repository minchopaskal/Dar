#pragma once

#include "fps_camera_controller.h"

struct Camera;

struct FPSEditModeCameraController : public FPSCameraController {
	FPSEditModeCameraController(Camera *cam, double movementSpeed);

	// Inherited via FPSCameraController
	void onMouseMove(double xPos, double yPos, double deltaTime) override final;

	void processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltaTime) override final;

	void onDrawUI() override final;

private:
	bool flyMode;
};
