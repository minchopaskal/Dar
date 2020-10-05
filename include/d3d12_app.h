#pragma once

#include <D3d12.h>
#include <dxgi.h>
#include <dxgi1_4.h>

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

struct D3D12App {
	D3D12App(UINT width, UINT height, const char *windowTitle) :
		width(width),
		height(height) { 
		strncpy(title, windowTitle, strlen(windowTitle) + 1);
		title[strlen(windowTitle)] = '\0';
	}

	const char* getTitle() const {
		return title;
	}

	UINT getWidth() const {
		return width;
	}

	UINT getHeight() const {
		return height;
	}

	virtual bool init() = 0;
	virtual void deinit() = 0;
	virtual void update() = 0;
	virtual void render() = 0;

protected:
	char title[256];
	UINT width, height;
};

