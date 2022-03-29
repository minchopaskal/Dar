#include "fps_camera_controller.h"
#include "framework/input_query.h"

#include "imgui.h"
#include "GLFW/glfw3.h"

FPSCameraController::FPSCameraController(Camera *cam, double movementSpeed) :
	ICameraController(cam),
	window(nullptr),
	mousePos{ 0.f, 0.f },
	speed(movementSpeed),
	mouseSensitivity(0.1f),
	mousePosValid(false) { }

void FPSCameraController::onMouseMove(double xPos, double yPos, double deltaTime) {
	if (!mousePosValid) {
		mousePos.x = xPos;
		mousePos.y = yPos;
		mousePosValid = true;
		return;
	}

	const double xOffset = (xPos - mousePos.x) * mouseSensitivity;
	const double yOffset = (yPos - mousePos.y) * mouseSensitivity;
	mousePos.x = xPos;
	mousePos.y = yPos;

	cam->yaw(xOffset);
	cam->pitch(yOffset);
}

double calculateZoomFactor(double scrollOffset, double deltaTime) {
	static const float ZOOM_SENSITIVITY = 50.f;
	
	double amount = ZOOM_SENSITIVITY * deltaTime;;
	return scrollOffset > 0.f ? 1.f + amount : 1.f - amount;
}

void FPSCameraController::onMouseScroll(double xOffset, double yOffset, double deltaTime) {
	cam->zoom(calculateZoomFactor(yOffset, deltaTime));
}

void FPSCameraController::processKeyboardInput(IKeyboardInputQuery *inputQuery, double deltaTime) {
	if (inputQuery == nullptr) {
		return;
	}

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

	if (inputQuery->query(GLFW_KEY_K).pressed) {
		cam->setKeepXZPlane(!cam->getKeepXZPlane());
	}

	ButtonState shiftState = inputQuery->query(GLFW_KEY_LEFT_SHIFT);
	if (shiftState.pressed && !shiftPressed) {
		shiftPressed = true;
		speed *= 2;
	}

	if (shiftState.released && shiftPressed) {
		shiftPressed = false;
		speed /= 2;
	}

	if (!shiftPressed) {
		const float deltaSpeed = 2 * speed * deltaTime;
		if (inputQuery->query(GLFW_KEY_T).pressed) {
			speed -= deltaSpeed;
		}

		if (inputQuery->query(GLFW_KEY_R).pressed) {
			speed += deltaSpeed;
		}
	}
}
