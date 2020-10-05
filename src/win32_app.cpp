#include "win32_app.h"

#include "d3d12_app.h"
#include <cstdio>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>
HWND Win32App::window = nullptr;

int Win32App::Run(D3D12App *app, HINSTANCE hInstance, int cmdShow) {
	WNDCLASSEX wc = { 0 };
	wc.lpfnWndProc = WindowProc;
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.hInstance = hInstance;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.lpszClassName = app->getTitle();
	RegisterClassEx(&wc);

	RECT windowRect = { 0, 0, static_cast<LONG>(app->getWidth()), static_cast<LONG>(app->getHeight()) };
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

	window = CreateWindowEx(
		0,
		app->getTitle(),
		app->getTitle(),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr,
		nullptr,
		hInstance,
		app
	);

	if (window == nullptr) {
		return 0;
	}

	if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
		AllocConsole(); // if a parent console didn't called us, create a new console
	
		FILE *dummyFile;
		freopen_s(&dummyFile, "nul", "w", stdout);

		HANDLE stdHandle = GetStdHandle(STD_ERROR_HANDLE);
		if (stdHandle != INVALID_HANDLE_VALUE) {
			int fileDescriptor = _open_osfhandle((intptr_t)stdHandle, _O_TEXT);
			if (fileDescriptor != -1) {
				FILE *file = _fdopen(fileDescriptor, "w");
				if (file != NULL) {
					int dup2Result = _dup2(_fileno(file), _fileno(stderr));
					if (dup2Result == 0) {
						setvbuf(stderr, NULL, _IONBF, 0);
					}
				}
			}
		}

		HANDLE stdHandle1 = GetStdHandle(STD_OUTPUT_HANDLE);
		if (stdHandle1 != INVALID_HANDLE_VALUE) {
			int fileDescriptor = _open_osfhandle((intptr_t)stdHandle1, _O_TEXT);
			if (fileDescriptor != -1) {
				FILE *file = _fdopen(fileDescriptor, "w");
				if (file != NULL) {
					int dup2Result = _dup2(_fileno(file), _fileno(stdout));
					if (dup2Result == 0) {
						setvbuf(stdout, NULL, _IONBF, 0);
					}
				}
			}
		}
	}

	const bool appInited = app->init();
	
	if (!appInited) {
		BOOL result = DestroyWindow(window);
		PostQuitMessage(result);
		return result;
	}

	ShowWindow(window, cmdShow);

	MSG msg = { };
	while (msg.message != WM_QUIT) {
		while (PeekMessage(&msg, window, 0, 0, PM_REMOVE) == TRUE) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	app->deinit();
	return msg.wParam;
}

HWND Win32App::getWindow() {
	return window;
}

LRESULT CALLBACK Win32App::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	D3D12App *app = reinterpret_cast<D3D12App*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

	LPCREATESTRUCT pCreateStruct;
	switch (uMsg) {
	case WM_CREATE:
		pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		//SetWindowLongPtr(hwnd, GWLP_USERDATA, lParam);
		return 0;
	case WM_PAINT:
		BeginPaint(hwnd, NULL);
		if (app) {
			app->update();
			app->render();
		}
		EndPaint(hwnd, NULL);
		return 0;
	case WM_DESTROY:
		BOOL result = DestroyWindow(hwnd);
		PostQuitMessage(result);
		return 0;
	}

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
