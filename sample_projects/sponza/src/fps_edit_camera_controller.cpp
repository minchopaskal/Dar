#include "fps_edit_camera_controller.h"
#include "d3d12_input_query.h"

#include "imgui.h"

FPSEditModeCameraController::FPSEditModeCameraController(Camera *cam, double movementSpeed) : 
	FPSCameraController(cam, movementSpeed),
	flyMode(false) 
{ }

void FPSEditModeCameraController::onMouseMove(double xPos, double yPos, double deltaTime) {
	if (!mousePosValid) {
		mousePos.x = xPos;
		mousePos.y = yPos;
		mousePosValid = true;
		return;
	}

	if (flyMode) {
		const double xOffset = (xPos - mousePos.x) * mouseSensitivity;
		const double yOffset = (yPos - mousePos.y) * mouseSensitivity;
		mousePos.x = xPos;
		mousePos.y = yPos;

		cam->yaw(xOffset);
		cam->pitch(yOffset);
	}
}

void setGLFWCursorHiddenState(GLFWwindow *window, bool show);

void FPSEditModeCameraController::processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltaTime) {
	if (inputQuery == nullptr) {
		return;
	}

	static bool leftAltPressed = false;

	ButtonState leftAltState = inputQuery->query(GLFW_KEY_LEFT_ALT);
	if (leftAltPressed != leftAltState.pressed) {
		leftAltPressed = !leftAltPressed;
		flyMode = leftAltPressed;
		setGLFWCursorHiddenState(window, flyMode == false);
	}

	if (flyMode) {
		FPSCameraController::processKeyboardInput(inputQuery, deltaTime);
	}
}

void FPSEditModeCameraController::onDrawUI() {
	ImGui::Begin("FPS Edit Mode Camera Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("[alt] - Hold for movement and rotation of camera");
	ImGui::Text("[mouse scroll] - Zoom/unzoom");
	ImGui::End();
}
