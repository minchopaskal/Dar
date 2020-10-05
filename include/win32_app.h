#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>

struct D3D12App;

struct Win32App {
	static int Run(D3D12App *app, HINSTANCE hInstance, int cmdShow);
	static HWND getWindow();

private:
	static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

private:
	static HWND window;
};