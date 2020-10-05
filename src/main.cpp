#include "d3d12_hello_triangle.h"
#include "win32_app.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int cmdShow) {
	D3D12HelloTriangle app(1280, 720, "Hello Triangle");

	return Win32App::Run(&app, hInstance, cmdShow);
}

