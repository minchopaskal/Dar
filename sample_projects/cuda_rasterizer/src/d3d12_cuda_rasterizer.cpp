#include "d3d12_cuda_rasterizer.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "framework/app.h"
#include "d3d12/resource_manager.h"
#include "asset_manager/asset_manager.h"
#include "utils/utils.h"

#include "cuda_manager.h"

// TODO: To make things simple, child projects should not rely on third party software.
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

#include "imgui.h"

CudaRasterizer::CudaRasterizer(Vector<String> &shaders, const String &windowTitle, UINT width, UINT height) :
	Dar::App(width, height, windowTitle.c_str()),
	aspectRatio(width / static_cast<float>(height)),
	cudaRenderTargetHost{ nullptr }
{
	inited = false;
	shaders.insert(shaders.begin(), "data\\rasterizer.ptx");
	shaders.insert(shaders.begin(), "data\\rasterizer_utils.ptx");
	if (!initializeCUDAManager(shaders, true)) {
		return;
	}
	cudaDevice = &getCUDAManager().getDevices()[0];

	// Switch to the context of this device from now on.
	if (cudaDevice->use().hasError()) {
		return;
	}

	inited = true;
}

CudaRasterizer::~CudaRasterizer() {
	deinit();
}

void CudaRasterizer::setUpdateFramebufferCallback(const UpdateFrameCallback cb, void *state) {
	updateFrameCb = cb;
	frameState = state;
}

void CudaRasterizer::setImGuiCallback(const DrawUICallback cb) {
	drawUICb = cb;
}

bool CudaRasterizer::isInitialized() const {
	return inited;
}

int CudaRasterizer::initImpl() {
	setUseImGui();

	return loadAssets();
}

void CudaRasterizer::deinit() {
	flush();
	deinitCUDAData();
	deinitializeCUDAManager();
	Super::deinit();
}

void CudaRasterizer::update() {
	if (updateFrameCb) {
		updateFrameCb(*this, frameState);
	}

	cudaDevice->use();

	const CUstream stream = cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution);
	cuStreamSynchronize(stream);

	cudaRenderTargetDevice.download(cudaRenderTargetHost);

	const Dar::UploadHandle handle = resManager->beginNewUpload();
	dx12RT.upload(handle, cudaRenderTargetHost);
	resManager->uploadBuffers();

	Dar::FrameData &fd = frameData[renderer.getBackbufferIndex()];
	fd.addTextureResource(dx12RT, 0);
	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0), 0);
}

void CudaRasterizer::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = std::max(1u, w);
	this->height = std::max(1u, h);
	aspectRatio = width / static_cast<float>(height);

	flush();

	renderer.resizeBackBuffers();

	initCUDAData();
}

void CudaRasterizer::onKeyboardInput(int key, int action) {
	
}

void CudaRasterizer::onMouseScroll(double xOffset, double yOffset) {
	
}

void CudaRasterizer::drawUI() {
	if (drawUICb) {
		drawUICb();
	}
}

Dar::FrameData &CudaRasterizer::getFrameData() {
	return frameData[renderer.getBackbufferIndex()];
}

CUDAError CudaRasterizer::setUseDepthBuffer(bool useDepthBuffer) {
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam<bool>(&useDepthBuffer, "useDepthBuffer"));

	CUDAMemHandle depthBufferHandle = NULL;
	if (useDepthBuffer) {
		if (depthBuffer.getSize() == static_cast<SizeType>(width) * height * sizeof(float)) {
			depthBufferHandle = depthBuffer.handle();
		} else {
			depthBuffer.initialize(static_cast<SizeType>(width) * height * sizeof(float));
			depthBufferHandle = depthBuffer.handle();
			clearDepthBuffer();
		}
	}

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&depthBufferHandle, "depthBuffer"));

	return CUDAError();
}

CUDAError CudaRasterizer::setVertexBuffer(const Vertex* buffer, SizeType verticesCount) {
	vertexBuffer.initialize(verticesCount * sizeof(Vertex));
	vertexBuffer.upload(buffer);

	auto *vs = new Vertex[verticesCount];
	cuMemcpyDtoH(reinterpret_cast<void*>(vs), vertexBuffer.handle(), vertexBuffer.getSize());

	const CUDAMemHandle vertexBufferHandle = vertexBuffer.handle();

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&vertexBufferHandle, "vertexBuffer"));
	
	cacheVerticesCount = verticesCount;

	return CUDAError();
}

CUDAError CudaRasterizer::setIndexBuffer(const unsigned int* buffer, SizeType indicesCount) {
	indexBuffer.initialize(indicesCount * sizeof(unsigned int));
	indexBuffer.upload(reinterpret_cast<const void*>(buffer));

	const CUDAMemHandle indexBufferHandle = indexBuffer.handle();

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&indexBufferHandle, "indexBuffer"));
	return CUDAError();
}

CUDAError CudaRasterizer::setUavBuffer(const void *buffer, SizeType size, int index) {
	if (index < 0 || index >= MAX_RESOURCES_COUNT) {
		return CUDAError(CUDA_ERROR_INVALID_VALUE, "Invalid UAV index", "Index of UAV resource is out of bounds!");
	}

	uavBuffers[index].initialize(size);
	uavBuffers[index].upload(buffer);

	const CUDAMemHandle uavBufferHandle = uavBuffers[index].handle();

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&uavBufferHandle, "resources", static_cast<SizeType>(index)));
	return CUDAError();
}

CUDAError CudaRasterizer::setShaderProgram(const String &name) const {
	CUDADefaultBuffer shaderPtrs;
	RETURN_ON_CUDA_ERROR_HANDLED(shaderPtrs.initialize(sizeof(CUDAShaderPointers)));

	CUDAFunction getShaderPtrsFunction(cudaDevice->getModule(), ("getShaderPtrs_" + name).c_str());
	getShaderPtrsFunction.addParams(
		shaderPtrs.handle()
	);
	RETURN_ON_CUDA_ERROR_HANDLED(getShaderPtrsFunction.launchSync(1, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	CUDAShaderPointers ptrs = { };
	RETURN_ON_CUDA_ERROR_HANDLED(shaderPtrs.download(&ptrs));
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&ptrs.vsShaderPtr, "vsShader"));
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&ptrs.psShaderPtr, "psShader"));

	return CUDAError();
}

CUDAError CudaRasterizer::drawIndexed(const unsigned int numPrimitives) const {
	constexpr unsigned int verticesInTriangle = 3;

	CUDAFunction vertexProcessingFunc(cudaDevice->getModule(), "processVertices");
	vertexProcessingFunc.addParams(
		cacheVerticesCount,
		width,
		height
	);
	RETURN_ON_CUDA_ERROR_HANDLED(vertexProcessingFunc.launchSync(numPrimitives * verticesInTriangle, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	CUDAFunction drawIndexedFunc(cudaDevice->getModule(), "drawIndexed");
	drawIndexedFunc.addParams(
		numPrimitives,
		width,
		height
	);
	
	RETURN_ON_CUDA_ERROR_HANDLED(drawIndexedFunc.launchSync(numPrimitives, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	return CUDAError();
}

CUDAError CudaRasterizer::setClearColor(const Vec4 &color) const {
	float colorHost[4];
	for (int i = 0; i < numComps; ++i) {
		colorHost[i] = color.data[i];
	}

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantArray(colorHost, 4, "clearColor"));

	return CUDAError();
}

CUDAError CudaRasterizer::setCulling(CudaRasterizerCullType cullType) const {
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&cullType, "cullType"));
	return CUDAError();
}

CUDAError CudaRasterizer::clearRenderTarget() {
	const CUDAMemHandle rt = cudaRenderTargetDevice.handle();
	RETURN_ON_CUDA_ERROR(cuMemsetD8(rt, 0, cudaRenderTargetDevice.getSize()));

	CUDAFunction blankKernel(cudaDevice->getModule(), "blank");
	blankKernel.addParams(
		rt,
		width,
		height
	);
	
	RETURN_ON_CUDA_ERROR_HANDLED(blankKernel.launchSync(width * height, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	return CUDAError();
}

CUDAError CudaRasterizer::clearDepthBuffer() {
	if (depthBuffer.getSize() == 0) {
		return CUDAError();
	}

	constexpr float floatOne = 1.f;
	const auto floatOneBits = *reinterpret_cast<const unsigned int*>(&floatOne);

	const CUDAMemHandle depthBufferHandle = depthBuffer.handle();
	RETURN_ON_CUDA_ERROR(cuMemsetD32(depthBufferHandle, floatOneBits, static_cast<SizeType>(width) * height));

	return CUDAError();
}

int CudaRasterizer::loadAssets() {
	// Create the pipeline state
	auto staticSamplerDesc = CD3DX12_STATIC_SAMPLER_DESC(0);

	Dar::PipelineStateDesc psDesc = {};
	psDesc.staticSamplerDesc = &staticSamplerDesc;
	psDesc.shaderName = L"screen_quad";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numTextures = 1;
	
	Dar::RenderPassDesc rpDesc = {};
	rpDesc.setPipelineStateDesc(psDesc);
	rpDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderer.addRenderPass(rpDesc);
	renderer.compilePipeline();

	// Create the texture resources that CUDA will use as render targets
	initCUDAData();

	return true;
}

void CudaRasterizer::deinitCUDAData() {
	dx12RT.deinit();
	cudaRenderTargetDevice.deinitialize();

	vertexBuffer.deinitialize();
	indexBuffer.deinitialize();

	depthBuffer.deinitialize();

	for (auto &uavBuffer : uavBuffers) {
		uavBuffer.deinitialize();
	}

	if (cudaRenderTargetHost != nullptr) {
		delete[] cudaRenderTargetHost;
		cudaRenderTargetHost = nullptr;
	}
}

void CudaRasterizer::initCUDAData() {
	deinitCUDAData();

	cudaDevice->use();

	Dar::TextureInitData resData = {};
	resData.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	resData.width = width;
	resData.height = height;
	resData.mipLevels = 1;
	dx12RT.init(resData, Dar::TextureResourceType::ShaderResource);
	dx12RT.setName(L"CUDARender");

	cudaRenderTargetHost = new float[static_cast<SizeType>(width) * height * numComps];
	
	cudaRenderTargetDevice.initialize(static_cast<SizeType>(width) * height * numComps * sizeof(float));
	clearRenderTarget();
	CUDAMemHandle rt = cudaRenderTargetDevice.handle();
	cudaDevice->uploadConstantParam(&rt, "renderTarget");

	cudaDevice->uploadConstantParam(&width, "width");
	cudaDevice->uploadConstantParam(&height, "height");
}
