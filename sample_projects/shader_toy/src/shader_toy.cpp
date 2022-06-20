#include "shader_toy.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "asset_manager/asset_manager.h"
#include "framework/app.h"
#include "d3d12/resource_manager.h"
#include "d3d12/pipeline_state.h"
#include "utils/utils.h"

// TODO: To make things simple, child projects should not rely on third party software
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

ShaderToy::ShaderToy(UINT width, UINT height, const String &windowTitle) : Dar::App(width, height, windowTitle.c_str()) { }

int ShaderToy::initImpl() {
	return loadAssets();
}

void ShaderToy::deinit() {
	flush();
	Super::deinit();
}

void ShaderToy::update() {
	auto &resManager = Dar::getResourceManager();

	Dar::UploadHandle uploadHandle = resManager.beginNewUpload();
	//resManager.uploadBufferData(uploadHandle, mvpResourceHandle[frameIndex], &MVP, sizeof(MVP));
	resManager.uploadBuffers();

	Dar::FrameData &fd = frameData[renderer.getBackbufferIndex()];
	fd.addRenderCommand(Dar::RenderCommand::drawInstanced(3, 1, 0, 0), 0);
}

void ShaderToy::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = dmath::max(1u, w);
	this->height = dmath::max(1u, h);

	flush();
}

void ShaderToy::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	auto &rs = renderer.getSettings();

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		rs.vSyncEnabled = !rs.vSyncEnabled;
	}
}

void ShaderToy::onMouseScroll(double xOffset, double yOffset) { }

Dar::FrameData &ShaderToy::getFrameData() {
	return frameData[renderer.getBackbufferIndex()];
}

int ShaderToy::loadAssets() {
	CD3DX12_STATIC_SAMPLER_DESC sampler{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR };

	Dar::PipelineStateDesc psDesc = {};
	psDesc.shaderName = L"screen_quad";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numRenderTargets = 1;
	psDesc.renderTargetFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	psDesc.staticSamplerDesc = &sampler;
	psDesc.cullMode = D3D12_CULL_MODE_NONE;

	Dar::RenderPassDesc renderPassDesc = {};
	renderPassDesc.setPipelineStateDesc(psDesc);
	renderPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderer.addRenderPass(renderPassDesc);

	renderer.compilePipeline();

	return true;
}
