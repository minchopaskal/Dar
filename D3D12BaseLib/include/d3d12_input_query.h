#pragma once

#include "GLFW/glfw3.h"

struct ButtonState {
	bool pressed;
	bool repeated;
	bool released;
	bool justPressed;
};

struct IKeyboardInputQuery {
	/// Query for keyboard key.
	/// @param key one of GLFW_KEY_* defines
	/// @return state of the queried key
	virtual ButtonState query(int key) = 0;
};
