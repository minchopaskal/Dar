#pragma once

#include "fps_camera_controller.h"

struct FPSEditModeCameraController : public FPSCameraController {
	FPSEditModeCameraController(Dar::Camera *cam, double movementSpeed);

	// Inherited via FPSCameraController
	void onMouseMove(double xPos, double yPos, double deltaTime) override final;

	void processKeyboardInput(Dar::IKeyboardInputQuery *inputQuery, double deltaTime) override final;

private:
	bool flyMode;
};
