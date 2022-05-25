#include "d3d12_tex_cube.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "framework/app.h"
#include "asset_manager/asset_manager.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/resource_manager.h"
#include "utils/utils.h"
#include "geometry.h"

// TODO: To make things simple, child projects should not rely on third party software
// Expose some input controller interface or something like that.
#include "GLFW/glfw3.h" // keyboard input

// For loading the texture image
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

#include "imgui.h"

struct ImageData {
	void *data = nullptr;
	int width = 0;
	int height = 0;
	int ncomp = 0;
};

ImageData loadImage(const wchar_t *name) {
	const WString pathStr = Dar::getAssetFullPath(name, Dar::AssetType::Texture);
	const wchar_t *path = pathStr.c_str();
	const SizeType bufferlen = wcslen(path) * sizeof(wchar_t);
	char *pathUtf8 = new char[bufferlen + 1];
	stbi_convert_wchar_to_utf8(pathUtf8, bufferlen, path);

	ImageData result = {};
	result.data = stbi_load(pathUtf8, &result.width, &result.height, nullptr, 4);
	result.ncomp = 4;

	return result;
}

D3D12TexturedCube::D3D12TexturedCube(const UINT w, const UINT h, const String &windowTitle) :
	Dar::App(w, h, windowTitle.c_str()),
	mvpBufferHandle{ INVALID_RESOURCE_HANDLE },
	camControl { &cam, 10.f }
{
	aspectRatio = static_cast<float>(width) / static_cast<float>(height);
	cam = Dar::Camera::perspectiveCamera(Vec3(0.f), 45.f, aspectRatio, 0.001f, 100.f);
}

int D3D12TexturedCube::initImpl() {
	setUseImGui();

	glfwSetInputMode(getGLFWWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!resizeDepthBuffer()) {
		return false;
	}

	return loadAssets();
}

void D3D12TexturedCube::deinit() {
	flush();
	Super::deinit();
}

void D3D12TexturedCube::update() {
	camControl.processKeyboardInput(this, deltaTime);

	// Update MVP matrices
	const auto angle = static_cast<float>(totalTime * 90.0);
	const auto rotationAxis = Vec3(0, 1, 1);
	auto modelMat = Mat4(1.f);
	modelMat = modelMat.rotate(rotationAxis, angle);
	modelMat = modelMat.translate({ 10.f, 0, 0.f });

	Mat4 viewMat = cam.getViewMatrix();
	Mat4 projectionMat = cam.getProjectionMatrix();

	Mat4 mvp = projectionMat * viewMat * modelMat;

	const int frameIndex = renderer.getBackbufferIndex();

	/// Initialize the MVP constant buffer resource if needed
	if (mvpBufferHandle[frameIndex] == INVALID_RESOURCE_HANDLE) {
		wchar_t frameMVPName[32] = L"";
		swprintf(frameMVPName, 32, L"MVPbuffer[%d]", frameIndex);

		Dar::ResourceInitData resData(Dar::ResourceType::DataBuffer);
		resData.size = sizeof(Mat4);
		resData.name = frameMVPName;
		mvpBufferHandle[frameIndex] = resManager->createBuffer(resData);
	}

	Dar::UploadHandle uploadHandle = resManager->beginNewUpload();
	resManager->uploadBufferData(uploadHandle, mvpBufferHandle[frameIndex], reinterpret_cast<void*>(&mvp), sizeof(Mat4));
	resManager->uploadBuffers();

	Dar::FrameData &fd = frameData[frameIndex];
	fd.addConstResource(mvpBufferHandle[frameIndex], 0);
	fd.setIndexBuffer(&indexBuffer);
	fd.setVertexBuffer(&vertexBuffer);
	
	for (int i = 0; i < numTextures; ++i) {
		fd.addTextureResource(textures[i], 0);
	}

	fd.addRenderCommand(Dar::RenderCommand::drawIndexedInstanced(36, 1, 0, 0, 0), 0);
}

void D3D12TexturedCube::drawUI() {
	ImGui::Begin("Stats");
	ImGui::Text("FPS: %.2f", fps);
	ImGui::Text("Camera FOV: %.2f", cam.getFOV());
	ImGui::Text("Camera speed: %.2f", camControl.getSpeed());
	Vec3 pos = cam.getPos();
	ImGui::Text("X: %.2f %.2f %.2f", pos.x, pos.y, pos.z);
	ImGui::Text("Camera Vectors:");
	Vec3 x = cam.getCameraX();
	Vec3 y = cam.getCameraY();
	Vec3 z = cam.getCameraZ();
	ImGui::Text("X: %.2f %.2f %.2f", x.x, x.y, x.z);
	ImGui::Text("Y: %.2f %.2f %.2f", y.x, y.y, y.z);
	ImGui::Text("Z: %.2f %.2f %.2f", z.x, z.y, z.z);
	ImGui::End();
}

void D3D12TexturedCube::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = dmath::max(1u, w);
	this->height = dmath::max(1u, h);
	aspectRatio = static_cast<float>(width) / static_cast<float>(height);
	cam.updateAspectRatio(width, height);

	flush();

	renderer.resizeBackBuffers();

	resizeDepthBuffer();
}

void D3D12TexturedCube::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}
	
	auto &rs = renderer.getSettings();

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		rs.vSyncEnabled = !rs.vSyncEnabled;
	}

	if (keyPressed[GLFW_KEY_O] && !keyRepeated[GLFW_KEY_O]) {
		projectionType = static_cast<ProjectionType>((static_cast<int>(projectionType) + 1) % 2);
	}
}

void D3D12TexturedCube::onMouseScroll(double xOffset, double yOffset) {
	camControl.onMouseScroll(xOffset, yOffset, deltaTime);
}

void D3D12TexturedCube::onMouseMove(double xPos, double yPos) {
	camControl.onMouseMove(xPos, yPos, deltaTime);
}

Dar::FrameData &D3D12TexturedCube::getFrameData() {
	const int frameIndex = renderer.getBackbufferIndex();

	return frameData[frameIndex];
}

int D3D12TexturedCube::loadAssets() {
	constexpr DXGI_FORMAT textureFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

	D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	CD3DX12_STATIC_SAMPLER_DESC sampler{ D3D12_FILTER_MIN_MAG_MIP_POINT };

	Dar::PipelineStateDesc psDesc = {};
	psDesc.shaderName = L"basic";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.inputLayouts = inputLayouts;
	psDesc.staticSamplerDesc = &sampler;
	psDesc.numInputLayouts = _countof(inputLayouts);
	psDesc.numConstantBufferViews = 1;
	psDesc.numTextures = numTextures;
	psDesc.depthStencilBufferFormat = DXGI_FORMAT_D32_FLOAT;

	Dar::RenderPassDesc renderPassDesc = {};
	renderPassDesc.setPipelineStateDesc(psDesc);
	renderPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderPassDesc.attach(Dar::RenderPassAttachment::depthStencil(&depthBuffer, true));
	renderer.addRenderPass(renderPassDesc);
	renderer.compilePipeline();

	/* Create and copy data to the vertex buffer*/
	{
		static Vertex cubeVertices[] = {
			{ {-1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f } }, // 0
			{ {-1.0f,  1.0f, -1.0f }, { 1.0f, 1.0f } }, // 1
			{ { 1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f } }, // 2
			{ { 1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f } }, // 3
			{ {-1.0f, -1.0f,  1.0f }, { 0.0f, 0.0f } }, // 4
			{ {-1.0f,  1.0f,  1.0f }, { 0.0f, 1.0f } }, // 5
			{ { 1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f } }, // 6
			{ { 1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f } }, // 7
			
			{ {-1.0f,  1.0f, -1.0f }, { 1.0f, 0.0f } }, // 8
			{ { 1.0f,  1.0f, -1.0f }, { 0.0f, 0.0f } }, // 9

			{ {-1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f } }, // 10
			{ { 1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f } }, // 11

			// plane below cube
			{ {-10.0f, -2.0f, -10.0f }, { 0.0f, 0.0f } }, // 12
			{ {-10.0f, -2.0f,  10.0f }, { 0.0f, 1.0f } }, // 13
			{ { 10.0f, -2.0f, -10.0f }, { 1.0f, 0.0f } }, // 14
			{ { 10.0f, -2.0f,  10.0f }, { 1.0f, 1.0f } }, // 15
		};
		constexpr UINT vertexBufferSize = sizeof(cubeVertices);

		static WORD cubeIndices[] = { 
			0, 1, 2, 0, 2, 3,
			4, 6, 5, 4, 7, 6,
			4, 5, 1, 4, 1, 0,
			3, 2, 6, 3, 6, 7,
			8, 5, 6, 1, 6, 9,
			4, 10, 11, 4, 11, 7,

			// plane
			12, 13, 15, 12, 15, 14
		};
		constexpr UINT indexBufferSize = sizeof(cubeIndices);

		auto &resManager = Dar::getResourceManager();
		Dar::UploadHandle uploadHandle = resManager.beginNewUpload();

		Dar::VertexIndexBufferDesc vertexDesc = {};
		vertexDesc.data = cubeVertices;
		vertexDesc.size = vertexBufferSize;
		vertexDesc.vertexBufferStride = sizeof(Vertex);
		vertexDesc.name = L"VertexBuffer";
		vertexBuffer.init(vertexDesc, uploadHandle);

		Dar::VertexIndexBufferDesc indexDesc = {};
		indexDesc.data = cubeIndices;
		indexDesc.size = indexBufferSize;
		indexDesc.indexBufferFormat = DXGI_FORMAT_R16_UINT;
		indexDesc.name = L"IndexBuffer";
		indexBuffer.init(indexDesc, uploadHandle);

		/* Load the texture */
		ImageData texData[numTextures];
		for (int i = 0; i < numTextures; ++i) {
			texData[i] = loadImage(L"box.jpg");
			wchar_t textureName[32] = L"";
			swprintf(textureName, 32, L"Texture[%d]", i);

			auto &tex = textures[i];
			Dar::TextureInitData texInitData = { };
			texInitData.width = texData->width;
			texInitData.height = texData->height;
			texInitData.format = textureFormat;
			tex.init(texInitData, Dar::TextureResourceType::ShaderResource);

			D3D12_SUBRESOURCE_DATA textureSubresources = {};
			textureSubresources.pData = texData[i].data;
			textureSubresources.RowPitch = static_cast<UINT64>(texData[i].width) * static_cast<UINT64>(texData[i].ncomp);
			textureSubresources.SlicePitch = textureSubresources.RowPitch * texData[0].height;
			resManager.uploadTextureData(uploadHandle, textures[i].getHandle(), &textureSubresources, 1, 0);
		}

		resManager.uploadBuffers();
	}

	return true;
}

bool D3D12TexturedCube::resizeDepthBuffer() {
	auto device = renderer.getDevice();
	return depthBuffer.init(device, width, height, DXGI_FORMAT_D32_FLOAT);
}