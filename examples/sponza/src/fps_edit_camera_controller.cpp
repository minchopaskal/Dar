#include "fps_edit_camera_controller.h"
#include "framework/input_query.h"

#include "imgui.h"

FPSEditModeCameraController::FPSEditModeCameraController(Dar::Camera *cam, double movementSpeed) :
	FPSCameraController(cam, movementSpeed),
	flyMode(false) 
{ }

void FPSEditModeCameraController::onMouseMove(double xPos, double yPos, double deltaTime) {
	if (flyMode) {
		FPSCameraController::onMouseMove(xPos, yPos, deltaTime);
	} else {
		mousePosValid = false;
	}
}

void setGLFWCursorHiddenState(GLFWwindow *window, bool show);

void FPSEditModeCameraController::processKeyboardInput(Dar::IKeyboardInputQuery *inputQuery, double deltaTime) {
	if (inputQuery == nullptr) {
		return;
	}

	static bool leftAltPressed = false;

	Dar::ButtonState leftAltState = inputQuery->query(GLFW_KEY_LEFT_ALT);
	if (leftAltPressed != leftAltState.pressed) {
		leftAltPressed = !leftAltPressed;
		flyMode = leftAltPressed;
		setGLFWCursorHiddenState(window, flyMode == false);
	}

	FPSCameraController::processKeyboardInput(inputQuery, deltaTime);
}
