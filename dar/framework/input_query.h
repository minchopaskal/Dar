#pragma once

#include "GLFW/glfw3.h"

namespace Dar {

struct ButtonState {
	bool pressed;
	bool repeated;
	bool released;
	bool justPressed;
};

struct IKeyboardInputQuery {
	/// Query for keyboard key's state.
	/// @param key one of GLFW_KEY_* defines
	/// @return state of the queried key
	virtual ButtonState query(int key) = 0;

	/// Query if key is pressed but not hold.
	/// @param key one of GLFW_KEY_* defines
	/// @return true if key was pressed once
	virtual bool queryPressed(int key) = 0;

	/// Query if key was just released.
	/// @param key one of GLFW_KEY_* defines
	/// @return true if key was just released.
	virtual bool queryReleased(int key) = 0;
};

} //namespace Dar