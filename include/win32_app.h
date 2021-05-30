#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>

#include <bitset>

struct D3D12App;
struct GLFWwindow;

// TODO: this is unnecessary. Either make it C-like API or a singleton.
struct Win32App {
	static int Run(D3D12App *app);
	static HWND getWindow();
	static D3D12App* getD3D12App();
	static void toggleFullscreen();

public:
	static constexpr int keysCount = 90; // see GLFW_KEY_Z
	static std::bitset<keysCount> keyPressed;
	static std::bitset<keysCount> keyRepeated;

	static bool vSyncEnabled;
	static bool tearingEnabled;

private:
	static D3D12App *app;
	static GLFWwindow *glfwWindow;
	static HWND window;
	static RECT windowRect;
	static bool fullscreen;
};