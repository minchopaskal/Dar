#pragma once

#include <Windows.h>

namespace D3D12 {

/// Class for measuring time in milliseconds.
struct Timer {
	Timer() : frequency(0.0) {
		startTime.QuadPart = 0;
		LARGE_INTEGER temp;
		QueryPerformanceFrequency(&temp);
		frequency = static_cast<double>(temp.QuadPart) / 1000.0;

		// Start the timer
		QueryPerformanceCounter(&startTime);
	}

	void restart() {
		QueryPerformanceCounter(&startTime);
	}

	/// Get time since the timer was launched or last restarted
	/// @return time since last launch in milliseconds
	float time() {
		LARGE_INTEGER endTime;
		QueryPerformanceCounter(&endTime);
		double elapsedTime = static_cast<double>(endTime.QuadPart) - static_cast<double>(startTime.QuadPart);

		return static_cast<float>(elapsedTime / frequency);
	}

private:
	LARGE_INTEGER startTime;
	double frequency;
};

} // namespace D3D12
