#include "hello_triangle.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "framework/app.h"
#include "d3d12/resource_manager.h"
#include "d3d12/pipeline_state.h"
#include "utils/utils.h"
#include "geometry.h"

#include "reslib/resource_library.h"

// TODO: To make things simple, child projects should not rely on third party software
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

HelloTriangle::HelloTriangle(UINT width, UINT height, const String &windowTitle) :
	Dar::App(width, height, windowTitle.c_str()),
	aspectRatio(width / float(height)),
	FOV(45.0) { }

bool HelloTriangle::initImpl() {
	if (!resizeDepthBuffer(this->width, this->height)) {
		return false;
	}

	renderer.init(device, true /* renderToScreen */);

	for (int i = 0; i < Dar::FRAME_COUNT; ++i) {
		mvpResource[i].init(sizeof(MVP), 1);
	}

	return loadAssets();
}

void HelloTriangle::deinit() {
	flush();
	Super::deinit();
}

void HelloTriangle::update() {
	// Update MVP matrices
	Mat4 modelMat = Mat4(1.f);

	float angle = glm::radians(static_cast<float>(getTotalTime()) * 90.f);
	const Vec3 rotationAxis = Vec3(0, 1, 0);
	modelMat = glm::rotate(modelMat, angle, rotationAxis);

	const Vec3 eyePosition = Vec3(0, 0, -5);
	const Vec3 focusPoint  = Vec3(0, 0, 0);
	const Vec3 upDirection = Vec3(0, 1, 0);
	Mat4 viewMat = glm::lookAt(eyePosition, focusPoint, upDirection);
	Mat4 projectionMat = glm::perspective(FOV, aspectRatio, 0.1f, 100.f);

	MVP = projectionMat * viewMat * modelMat;

	const int frameIndex = renderer.getBackbufferIndex();

	Dar::UploadHandle uploadHandle = resManager->beginNewUpload();
	resManager->uploadBufferData(uploadHandle, mvpResource[frameIndex].getHandle(), &MVP, sizeof(MVP));
	resManager->uploadBuffers();

	Dar::FrameData &fd = frameData[renderer.getBackbufferIndex()];
	fd.setVertexBuffer(&vertexBuffer);
	fd.setIndexBuffer(&indexBuffer);
	fd.addConstResource(mvpResource[frameIndex].getHandle(), 0);

	fd.startNewPass();
	fd.addRenderCommand(Dar::RenderCommandDrawIndexedInstanced(3, 1, 0, 0, 0));

	renderer.renderFrame(fd);
}

void HelloTriangle::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = glm::max(1u, w);
	this->height = glm::max(1u, h);
	aspectRatio = width / float(height);

	flush();

	device.resizeBackBuffers();

	resizeDepthBuffer(this->width, this->height);
}

void HelloTriangle::onKeyboardInput(int /*key*/, int /*action*/) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	auto &rs = renderer.getSettings();

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		rs.vSyncEnabled = !rs.vSyncEnabled;
	}

	if (keyPressed[GLFW_KEY_ESCAPE] && !keyRepeated[GLFW_KEY_ESCAPE]) {
		glfwSetWindowShouldClose(getGLFWWindow(), true);
	}
}

void HelloTriangle::onMouseScroll(double /*xOffset*/, double yOffset) {
	static const double speed = 500.f;
	FOV -= float(speed * getDeltaTime() * yOffset);
	FOV = glm::fclamp(FOV, 30.f, 120.f);
}

Dar::FrameData &HelloTriangle::getFrameData() {
	return frameData[renderer.getBackbufferIndex()];
}

void HelloTriangle::beginFrame() {
	App::beginFrame();

	renderer.beginFrame();
	getFrameData().beginFrame(renderer);
}

void HelloTriangle::endFrame() {
	getFrameData().endFrame(renderer);
	renderer.endFrame();

	App::endFrame();
}

int HelloTriangle::loadAssets() {
	auto &resLibrary = Dar::getResourceLibrary();
	resLibrary.LoadShaderData();

	CD3DX12_STATIC_SAMPLER_DESC sampler{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR };
	D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	Dar::PipelineStateDesc psDesc = {};
	psDesc.shaderName = "basic";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numRenderTargets = 1;
	psDesc.renderTargetFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	psDesc.inputLayouts = inputLayouts;
	psDesc.numInputLayouts = _countof(inputLayouts);
	psDesc.staticSamplerDescs = &sampler;
	psDesc.numStaticSamplers = 1;
	psDesc.cullMode = D3D12_CULL_MODE_NONE;
	psDesc.depthStencilBufferFormat = depthBuffer.getFormatAsDepthBuffer();
	psDesc.numConstantBufferViews = 1;

	Dar::RenderPassDesc renderPassDesc = {};
	renderPassDesc.setPipelineStateDesc(psDesc);
	renderPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderPassDesc.attach(Dar::RenderPassAttachment::depthStencil(&depthBuffer, true));
	framePipeline.addRenderPass(renderPassDesc);
	framePipeline.compilePipeline(device);

	renderer.setFramePipeline(&framePipeline);

	/* Create and copy data to the vertex buffer*/
	{
		static Vertex triangleVertices[] = {
			{ {   0.0f,  0.5f * aspectRatio, 0.0f, 1.f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
			{ {  0.5f, -0.5f * aspectRatio, 0.0f, 1.f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
			{ { -0.5f, -0.5f * aspectRatio, 0.0f, 1.f }, { 0.0f, 0.0f, 1.0f, 1.0f } }
		};
		const UINT vertexBufferSize = sizeof(triangleVertices);

		static WORD triangleIndices[] = { 0, 1, 2 };
		const UINT indexBufferSize = sizeof(triangleIndices);


		Dar::UploadHandle uploadHandle = resManager->beginNewUpload();

		Dar::VertexIndexBufferDesc vertexDesc = {};
		vertexDesc.data = triangleVertices;
		vertexDesc.name = "VertexBuffer";
		vertexDesc.size = vertexBufferSize;
		vertexDesc.vertexBufferStride = sizeof(Vertex);
		vertexBuffer.init(vertexDesc, uploadHandle);

		Dar::VertexIndexBufferDesc indexDesc = {};
		indexDesc.data = triangleIndices;
		indexDesc.size = indexBufferSize;
		indexDesc.name = "IndexBuffer";
		indexDesc.indexBufferFormat = DXGI_FORMAT_R16_UINT;
		indexBuffer.init(indexDesc, uploadHandle);

		resManager->uploadBuffers();
	}

	return true;
}

bool HelloTriangle::resizeDepthBuffer(int w, int h) {
	return depthBuffer.init(device.getDevice(), w, h, DXGI_FORMAT_D32_FLOAT, "DepthBuffer");
}
