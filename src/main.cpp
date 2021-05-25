#include "d3d12_hello_triangle.h"
#include "win32_app.h"

D3D12HelloTriangle app(1280, 720, "Hello Triangle");
HWND hGLFWWinWnd;

int main(int argc, char **argv) {
	D3D12HelloTriangle app(1280, 720, "Hello Triangle");
	Win32App::Run(&app);

	return 0;
}
