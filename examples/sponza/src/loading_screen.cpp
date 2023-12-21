#include "loading_screen.h"

#include "framework/app.h"

void LoadingScreen::init(Dar::Device &device) {
	LOG_FMT(Info, "LoadingScreen::init");

	renderer.init(device, true /* renderToScreen */);

	D3D12_STATIC_SAMPLER_DESC samplers[] = {
		CD3DX12_STATIC_SAMPLER_DESC{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR },
	};

	Dar::RenderPassDesc passDesc = {};
	Dar::PipelineStateDesc psDesc = {};
	psDesc.shaderName = "loading_screen";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numConstantBufferViews = 1;
	psDesc.cullMode = D3D12_CULL_MODE_NONE;
	passDesc.setPipelineStateDesc(psDesc);
	passDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());

	pipeline.init();
	pipeline.addRenderPass(passDesc);
	pipeline.compilePipeline(device);

	renderer.setFramePipeline(&pipeline);

	auto &resManager = Dar::getResourceManager();
	for (int i = 0; i < Dar::FRAME_COUNT; ++i) {
		char frameConstDataName[32] = "";
		snprintf(frameConstDataName, 32, "LoadingScreenConstData[%d]", i);

		Dar::ResourceInitData resData(Dar::ResourceType::DataBuffer);
		resData.size = sizeof(LoadingScreenConstData);
		resData.name = frameConstDataName;
		constDataHandles[i] = resManager.createBuffer(resData);
	}

	LOG_FMT(Info, "LoadingScreen::init SUCCESS");
}

void LoadingScreen::deinit() {
	pipeline.deinit();
	renderer.deinit();
}

void LoadingScreen::beginFrame() {
	renderer.beginFrame();
	frameData[renderer.getBackbufferIndex()].beginFrame(renderer);
}

void LoadingScreen::endFrame() {
	frameData[renderer.getBackbufferIndex()].endFrame(renderer);
	renderer.endFrame();
}

void LoadingScreen::render() {
	auto app = Dar::getApp();
	timePassed += app->getDeltaTime();

	uploadConstData();

	const int frameIndex = renderer.getBackbufferIndex();
	Dar::FrameData& fd = frameData[frameIndex];
	fd.addConstResource(constDataHandles[frameIndex], 0);

	fd.startNewPass();
	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0));
	if (renderer.getNumRenderedFrames() < Dar::FRAME_COUNT) {
		fd.setUseSameCommands(true);
	}

	renderer.renderFrame(fd);
}

void LoadingScreen::uploadConstData() {
	auto app = Dar::getApp();

	LoadingScreenConstData constData = {};
	constData.width = app->getWidth();
	constData.height = app->getHeight();
	constData.time = static_cast<float>(timePassed);
	constData.delta = static_cast<float>(app->getDeltaTime());

	// TODO: Shader model 6.0+ supports uint64_t but do we really need it?
	// Most probably it's fine when overflow happens here as frame count
	// will be used in some periodic function.
	constData.frame = static_cast<UINT>(renderer.getNumRenderedFrames());

	/// Initialize the MVP constant buffer resource if needed
	const int frameIndex = renderer.getBackbufferIndex();

	auto &resManager = Dar::getResourceManager();
	Dar::UploadHandle uploadHandle = resManager.beginNewUpload();
	resManager.uploadBufferData(uploadHandle, constDataHandles[frameIndex], reinterpret_cast<void*>(&constData), sizeof(LoadingScreenConstData));
	resManager.uploadBuffers();
}
