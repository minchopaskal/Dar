#include "fps_camera_controller.h"
#include "framework/input_query.h"

FPSCameraController::FPSCameraController(Dar::Camera *cam, double movementSpeed) : 
	ICameraController(cam),
	mousePos{ 0.f, 0.f },
	speed(movementSpeed),
	mouseSensitivity(0.1f),
	mousePosValid(false)
{
	//cam->setKeepXZPlane(true);
}

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
	if (inputQuery->query('w').pressed) {
		cam->moveForward(amount);
	}

	if (inputQuery->query('s').pressed) {
		cam->moveForward(-amount);
	}

	if (inputQuery->query('d').pressed) {
		cam->moveRight(amount);
	}

	if (inputQuery->query('a').pressed) {
		cam->moveRight(-amount);
	}

	if (inputQuery->query('e').pressed) {
		cam->moveUp(amount);
	}

	if (inputQuery->query('q').pressed) {
		cam->moveUp(-amount);
	}


	const float deltaSpeed = 2 * speed * deltaTime;
	if (inputQuery->query('r').pressed) {
		speed -= deltaSpeed;
	}

	if (inputQuery->query('t').pressed) {
		speed += deltaSpeed;
	}
}
