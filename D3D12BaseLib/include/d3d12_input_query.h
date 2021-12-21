#pragma once

struct ButtonState {
	bool pressed;
	bool repeated;
};

struct IKeyboardInputQuery {
	/// Query for keyboard key.
	/// @param key corresponds ASCII character
	/// @return state of the queried key
	virtual ButtonState query(char key) = 0;
};
