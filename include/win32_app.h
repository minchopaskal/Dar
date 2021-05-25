#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>

struct D3D12App;
struct GLFWwindow;

struct Win32App {
	static int Run(D3D12App *app);
	static HWND getWindow();
	static D3D12App* getApp();

private:
	static D3D12App *app;
	static HWND window;
};