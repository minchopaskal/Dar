#include "d3d12_hello_triangle.h"

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
#include "geometry.h"

// TODO: To make things simple, child projects should not rely on third party software
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

D3D12HelloTriangle::D3D12HelloTriangle(UINT width, UINT height, const String &windowTitle) :
	Dar::App(width, height, windowTitle.c_str()),
	aspectRatio(width / float(height)),
	FOV(45.0) { }

int D3D12HelloTriangle::initImpl() {
	if (!resizeDepthBuffer(this->width, this->height)) {
		return false;
	}

	return loadAssets();
}

void D3D12HelloTriangle::deinit() {
	flush();
	Super::deinit();
}

void D3D12HelloTriangle::update() {
	// Update MVP matrices
	//float angle = static_cast<float>(totalTime * 90.0);
	//const Vec3 rotationAxis = Vec3(0, 1, 0);
	Mat4 modelMat = Mat4(1.f);
	//modelMat = modelMat.rotate(rotationAxis, angle);
	//modelMat = modelMat.translate({ 1, 0, 0 });

	const Vec3 eyePosition = Vec3(0, 0, -10);
	const Vec3 focusPoint  = Vec3(0, 0, 0);
	const Vec3 upDirection = Vec3(0, 1, 0);
	Mat4 viewMat = dmath::lookAt(focusPoint, eyePosition, upDirection);
	Mat4 projectionMat = dmath::perspective(FOV, aspectRatio, 0.1f, 100.f);

	MVP = projectionMat * viewMat * modelMat;

	auto &resManager = Dar::getResourceManager();
	const int frameIndex = renderer.getBackbufferIndex();
	if (mvpResourceHandle[frameIndex] == INVALID_RESOURCE_HANDLE) {
		Dar::ResourceInitData resInitData(Dar::ResourceType::DataBuffer);
		resInitData.size = sizeof(MVP);
		resInitData.name = L"MVP matrix";
		mvpResourceHandle[frameIndex] = resManager.createBuffer(resInitData);
	}

	Dar::UploadHandle uploadHandle = resManager.beginNewUpload();
	resManager.uploadBufferData(uploadHandle, mvpResourceHandle[frameIndex], &MVP, sizeof(MVP));
	resManager.uploadBuffers();

	Dar::FrameData &fd = frameData[renderer.getBackbufferIndex()];
	fd.setVertexBuffer(&vertexBuffer);
	fd.setIndexBuffer(&indexBuffer);
	fd.addConstResource(mvpResourceHandle[frameIndex], 0);
	fd.addRenderCommand(Dar::RenderCommand::drawIndexedInstanced(3, 1, 0, 0, 0), 0);
}

void D3D12HelloTriangle::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = dmath::max(1u, w);
	this->height = dmath::max(1u, h);
	aspectRatio = width / float(height);

	flush();

	renderer.resizeBackBuffers();

	resizeDepthBuffer(this->width, this->height);
}

void D3D12HelloTriangle::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	auto &rs = renderer.getSettings();

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		rs.vSyncEnabled = !rs.vSyncEnabled;
	}
}

void D3D12HelloTriangle::onMouseScroll(double xOffset, double yOffset) {
	static const double speed = 500.f;
	FOV -= float(speed * deltaTime * yOffset);
	FOV = dmath::min(dmath::max(30.f, FOV), 120.f);
}

Dar::FrameData &D3D12HelloTriangle::getFrameData() {
	return frameData[renderer.getBackbufferIndex()];
}

int D3D12HelloTriangle::loadAssets() {
	CD3DX12_STATIC_SAMPLER_DESC sampler{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR };
	D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	Dar::PipelineStateDesc psDesc = {};
	psDesc.shaderName = L"basic";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numRenderTargets = 1;
	psDesc.renderTargetFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	psDesc.inputLayouts = inputLayouts;
	psDesc.numInputLayouts = _countof(inputLayouts);
	psDesc.staticSamplerDesc = &sampler;
	psDesc.depthStencilBufferFormat = depthBuffer.getFormatAsDepthBuffer();
	psDesc.numConstantBufferViews = 1;

	Dar::RenderPassDesc renderPassDesc = {};
	renderPassDesc.setPipelineStateDesc(psDesc);
	renderPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderPassDesc.attach(Dar::RenderPassAttachment::depthStencil(&depthBuffer, true));
	renderer.addRenderPass(renderPassDesc);

	renderer.compilePipeline();

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
		vertexDesc.name = L"VertexBuffer";
		vertexDesc.size = vertexBufferSize;
		vertexDesc.vertexBufferStride = sizeof(Vertex);
		vertexBuffer.init(vertexDesc, uploadHandle);

		Dar::VertexIndexBufferDesc indexDesc = {};
		indexDesc.data = triangleIndices;
		indexDesc.size = indexBufferSize;
		indexDesc.name = L"IndexBuffer";
		indexDesc.indexBufferFormat = DXGI_FORMAT_R16_UINT;
		indexBuffer.init(indexDesc, uploadHandle);

		resManager->uploadBuffers();
	}

	return true;
}

bool D3D12HelloTriangle::resizeDepthBuffer(int width, int height) {
	return depthBuffer.init(renderer.getDevice(), width, height, DXGI_FORMAT_D32_FLOAT);
}
