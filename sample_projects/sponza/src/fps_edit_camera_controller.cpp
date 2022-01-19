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
		double amount = speed * deltaTime;
		if (inputQuery->query(GLFW_KEY_W).pressed) {
			cam->moveForward(amount);
		}

		if (inputQuery->query(GLFW_KEY_S).pressed) {
			cam->moveForward(-amount);
		}

		if (inputQuery->query(GLFW_KEY_D).pressed) {
			cam->moveRight(amount);
		}

		if (inputQuery->query(GLFW_KEY_A).pressed) {
			cam->moveRight(-amount);
		}

		if (inputQuery->query(GLFW_KEY_E).pressed) {
			cam->moveUp(amount);
		}

		if (inputQuery->query(GLFW_KEY_Q).pressed) {
			cam->moveUp(-amount);
		}

		const float deltaSpeed = 2 * speed * deltaTime;
		if (inputQuery->query(GLFW_KEY_T).pressed) {
			speed -= deltaSpeed;
		}

		if (inputQuery->query(GLFW_KEY_R).pressed) {
			speed += deltaSpeed;
		}
	}
}

void FPSEditModeCameraController::onDrawUI() {
	ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Text("[alt] - Hold for movement and rotation of camera");
	ImGui::Text("[mouse scroll] - Zoom/unzoom");
	ImGui::Text("[alt + wasd] - Move forwards/left/backwards/right");
	ImGui::Text("[alt + qe] - Move up/down");
	ImGui::Text("[alt + rt] - Increase/Decrease camera speed");
	ImGui::Text("[m] - Switch to FPS mode"); // TODO: change controller here. When mouse is shown enter edit mode.
	ImGui::End();
}
