#pragma once

#include <bitset>
#include <string>

#include "d3d12_app.h"
#include "defines.h"

struct D3D12HelloTriangle : D3D12App {
	D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle);

	// Inherited via D3D12App
	virtual int init() override;
	virtual int loadAssets() override;
	virtual void deinit() override;
	virtual void update() override;
	virtual void render() override;
	virtual void resize(int width, int height) override;
	virtual void keyboardInput(int key, int action) override;

private:
	ComPtr<ID3D12GraphicsCommandList2> populateCommandList();

	void timeIt();

private:
	ComPtr<ID3D12RootSignature> rootSignature;
	ComPtr<ID3D12PipelineState> pipelineState;

	// Vertex buffer
	ComPtr<ID3D12Resource> vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	real aspectRatio;

	// Input
	static const int keysCount = 90; // see GLFW_KEY_Z
	std::bitset<keysCount> keyPressed;
	std::bitset<keysCount> keyRepeated;

	// timing
	double fps;
};
