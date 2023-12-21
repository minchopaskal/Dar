#include "fps_camera_controller.h"
#include "framework/input_query.h"

#include "imgui.h"
#include "GLFW/glfw3.h"

FPSCameraController::FPSCameraController(Dar::Camera *cam, double movementSpeed) :
	Dar::ICameraController(cam),
	window(nullptr),
	mousePos{ 0.f, 0.f },
	speed(movementSpeed),
	mouseSensitivity(0.1f),
	mousePosValid(false) { }

void FPSCameraController::onMouseMove(double xPos, double yPos, double /*deltaTime*/) {
	if (mouseDisabled) {
		return;
	}

	if (!mousePosValid) {
		mousePos.x = xPos;
		mousePos.y = yPos;
		mousePosValid = true;
		return;
	}

	const float xOffset = static_cast<float>((xPos - mousePos.x) * mouseSensitivity);
	const float yOffset = static_cast<float>((yPos - mousePos.y) * mouseSensitivity);
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

void FPSCameraController::onMouseScroll(double /*xOffset*/, double yOffset, double deltaTime) {
	cam->zoom(static_cast<float>(calculateZoomFactor(yOffset, deltaTime)));
}

void FPSCameraController::processKeyboardInput(Dar::IKeyboardInputQuery *inputQuery, double deltaTime) {
	if (inputQuery == nullptr) {
		return;
	}

	float amount = static_cast<float>(speed * deltaTime);
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

	Dar::ButtonState shiftState = inputQuery->query(GLFW_KEY_LEFT_SHIFT);
	if (shiftState.pressed && !shiftPressed) {
		shiftPressed = true;
		speed *= 2;
	}

	if (shiftState.released && shiftPressed) {
		shiftPressed = false;
		speed /= 2;
	}

	if (!shiftPressed) {
		const float deltaSpeed = static_cast<float>(2 * speed * deltaTime);
		if (inputQuery->query(GLFW_KEY_T).pressed) {
			speed -= deltaSpeed;
		}

		if (inputQuery->query(GLFW_KEY_R).pressed) {
			speed += deltaSpeed;
		}
	}
}
