#include "hud.h"
#include "framework/app.h"
#include "framework/camera.h"

bool HUD::init(Dar::Device &device) {
	auto app = Dar::getApp();

	Dar::TextureInitData rtvTextureDesc = {};
	rtvTextureDesc.width = app->getWidth();
	rtvTextureDesc.height = app->getHeight();
	rtvTextureDesc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
	rtvTextureDesc.clearValue.color[0] = 0.f;
	rtvTextureDesc.clearValue.color[1] = 0.f;
	rtvTextureDesc.clearValue.color[2] = 0.f;
	rtvTextureDesc.clearValue.color[3] = 0.f;
	renderTarget.init(rtvTextureDesc, Dar::FRAME_COUNT, "HUD_RT");

	D3D12_INPUT_ELEMENT_DESC hudInputLayout[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
	};

	D3D12_STATIC_SAMPLER_DESC staticSampler = CD3DX12_STATIC_SAMPLER_DESC{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR };

	framePipeline.init();
	Dar::RenderPassDesc passDesc = {};
	Dar::PipelineStateDesc psDesc = {};
	psDesc.shaderName = "hud";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numConstantBufferViews = static_cast<unsigned int>(ConstantBuffers::Count);
	psDesc.numStaticSamplers = 1;
	psDesc.staticSamplerDescs = &staticSampler;
	psDesc.numRenderTargets = 1;
	psDesc.inputLayouts = hudInputLayout;
	psDesc.numInputLayouts = _countof(hudInputLayout);
	psDesc.cullMode = D3D12_CULL_MODE_NONE;
	passDesc.setPipelineStateDesc(psDesc);
	passDesc.attach(Dar::RenderPassAttachment::renderTarget(&renderTarget));
	framePipeline.addRenderPass(passDesc);
	framePipeline.compilePipeline(device);

	if (!hudRenderer.init(device, false /* renderToScreen */)) {
		LOG(Error, "Failed to initialize HUD renderer!");
		return false;
	}
	hudRenderer.setFramePipeline(&framePipeline);

	auto& resManager = Dar::getResourceManager();
	for (int i = 0; i < Dar::FRAME_COUNT; ++i) {
		char name[32];
		snprintf(name, 32, "HUDConstData[%d]", i);

		Dar::ResourceInitData initData(Dar::ResourceType::DataBuffer);
		initData.name = name;
		initData.size = sizeof(HUDConstData);

		constData[i] = resManager.createBuffer(initData);
	}

	return true;
}

void HUD::beginFrame() {
	for (int i = 0; i < MAX_DEPTH; ++i) {
		textWidgets[i].clear();
		rectWidgets[i].clear();
	}
	buttons.clear();

	hudRenderer.beginFrame();
	frameData[hudRenderer.getBackbufferIndex()].beginFrame(hudRenderer);
}

void HUD::endFrame() {
	frameData[hudRenderer.getBackbufferIndex()].endFrame(hudRenderer);
	hudRenderer.endFrame();
}

WidgetHandle HUD::addRectWidget(const RectWidgetDesc &rect) {
	if (rect.depth >= MAX_DEPTH) {
		dassert(false);
		return INVALID_WIDGET_HANDLE;
	}

	rectWidgets[rect.depth].push_back(rect);

	return static_cast<int>(WidgetType::Rect) * WIDGET_TYPE_OFFSET + (rect.depth * MAX_WIDGETS_PER_DEPTH + rectWidgets[rect.depth].size() - 1);
}

WidgetHandle HUD::addButton(const RectWidgetDesc & rect) {
	auto handle = addRectWidget(rect);

	buttons.insert(handle);

	return handle;
}

bool HUD::isOverButton(WidgetHandle handle, Vec2 pos) const {
	if (!buttons.contains(handle)) {
		return false;
	}

	handle = handle - static_cast<int>(WidgetType::Rect) * WIDGET_TYPE_OFFSET;
	auto depth = handle / MAX_WIDGETS_PER_DEPTH;
	auto index = handle - depth * MAX_WIDGETS_PER_DEPTH;
	auto& rect = rectWidgets[depth][index];
	return
		pos.x >= rect.topLeft.x
		&& pos.x <= rect.topLeft.x + rect.size.x
		&& pos.y >= rect.topLeft.y
		&& pos.y <= rect.topLeft.y + rect.size.y;
}

WidgetType HUD::getWidgetType(WidgetHandle handle) {
	for (int i = 0; i < static_cast<int>(WidgetType::WidgetCount) - 1; ++i) {
		if (handle >= i * WIDGET_TYPE_OFFSET && handle < (i + 1) * WIDGET_TYPE_OFFSET) {
			return static_cast<WidgetType>(i);
		}
	}
	
	return WidgetType::InvalidWidget;
}

FenceValue HUD::render() {
	Vector<HUDVertex> vertices;
	Vector<uint32_t> indices;
	Vector<WidgetData> widgetDatas;

	Vector<TextureDesc> textureDescs;
	Map<String, TextureId> currentMap;

	// Check for missing textures
	bool texturesNeedUpdate = false;
	for (int i = MAX_DEPTH - 1; i >= 0; --i) {
		for (auto &rect : rectWidgets[i]) {
			auto name = rect.texture.value_or("");
			if (name.empty()) {
				continue;
			}
			textureDescs.push_back(TextureDesc{ name, static_cast<unsigned int>(textureDescs.size()) });
			currentMap[name] = textureDescs.back().id;

			if (texturesNeedUpdate) {
				continue;
			}

			if (auto it = textureName2Id.find(name); it == textureName2Id.end()) {
				texturesNeedUpdate = true;
			}
		}
	}

	for (int i = MAX_DEPTH - 1; i >= 0; --i) {
		for (auto &rect : rectWidgets[i]) {
			const uint32_t startIndex = static_cast<uint32_t>(vertices.size());
			vertices.push_back(HUDVertex{ rect.topLeft, Vec2{0.f, 0.f} });
			vertices.push_back(HUDVertex{ Vec2{ rect.topLeft.x + rect.size.x, rect.topLeft.y }, Vec2{1.f, 0.f} });
			vertices.push_back(HUDVertex{ rect.topLeft + rect.size, Vec2{1.f, 1.f} });
			vertices.push_back(HUDVertex{ Vec2{ rect.topLeft.x, rect.topLeft.y + rect.size.y }, Vec2{0.f, 1.f} });

			indices.push_back(startIndex + 0);
			indices.push_back(startIndex + 1);
			indices.push_back(startIndex + 3);

			indices.push_back(startIndex + 1);
			indices.push_back(startIndex + 2);
			indices.push_back(startIndex + 3);

			int textureIndex = -1;
			if (rect.texture.has_value()) {
				if (texturesNeedUpdate) {
					textureIndex = currentMap[rect.texture.value()];
				} else {
					textureIndex = textureName2Id[rect.texture.value()];
				}
			}

			widgetDatas.push_back(WidgetData{
				.color = rect.color.value_or(Vec4{1.f, 0.f, 1.f, 1.f}),
				.textureIndex = textureIndex,
				.type = WidgetType::Rect,
			});
		}

		// TODO: text widgets
	}

	if (widgetDatas.empty()) {
		return true;
	}

	auto &resManager = Dar::getResourceManager();
	auto uploadHandle = resManager.beginNewUpload();

	if (texturesNeedUpdate) {
		if (!uploadTextureData(textureDescs, uploadHandle, textures, texturesHeap, true /* forceNoMips */)) {
			LOG(Error, "Failed to upload HUD textures!");
			return false;
		} else {
			textureName2Id = std::move(currentMap);
		}
	}

	if (!uploadConstData(uploadHandle)) {
		LOG(Error, "Failed to upload HUD const data!");
		return false;
	}

	if (!uploadWidgetData(widgetDatas, uploadHandle)) {
		LOG(Error, "Failed to upload HUD widget data!");
		return false;
	}

	const int frameIndex = hudRenderer.getBackbufferIndex();

	Dar::VertexIndexBufferDesc vertexDesc = {};
	vertexDesc.data = vertices.data();
	vertexDesc.size = static_cast<UINT>(vertices.size() * sizeof(HUDVertex));
	vertexDesc.name = "HUDVertexBuffer";
	vertexDesc.vertexBufferStride = sizeof(HUDVertex);
	if (!vertexBuffer[frameIndex].init(vertexDesc, uploadHandle)) {
		LOG(Error, "Failed to upload HUD vertex buffer!");
		return false;
	}

	Dar::VertexIndexBufferDesc indexDesc = {};
	indexDesc.data = indices.data();
	indexDesc.size = static_cast<UINT>(indices.size() * sizeof(uint32_t));
	indexDesc.name = "HUDIndexBuffer";
	indexDesc.indexBufferFormat = DXGI_FORMAT_R32_UINT;
	if (!indexBuffer[frameIndex].init(indexDesc, uploadHandle)) {
		LOG(Error, "Failed to upload HUD index buffer!");
		return false;
	}
	auto uploadCtx = resManager.uploadBuffersAsync();

	auto &fd = frameData[frameIndex];
	fd.addUploadContextToWait(uploadCtx);
	fd.addConstResource(constData[frameIndex], 0);

	fd.startNewPass();
	fd.setVertexBuffer(&vertexBuffer[frameIndex]);
	fd.setIndexBuffer(&indexBuffer[frameIndex]);
	fd.addDataBufferResource(widgetDataBuffer);
	for (auto &tex : textures) {
		fd.addTextureResource(tex);
	}
	fd.addRenderCommand(Dar::RenderCommandDrawIndexedInstanced{ static_cast<UINT>(indices.size()), 1, 0, 0, 0 });

	return hudRenderer.renderFrame(frameData[frameIndex]);
}

const Dar::TextureResource& HUD::getTexture() const {
	return renderTarget.getTextureResource(hudRenderer.getBackbufferIndex());
}

bool HUD::resize() {
	auto app = Dar::getApp();

	int w = app->getWidth();
	int h = app->getHeight();

	renderTarget.resizeRenderTarget(w, h);

	return true;
}

bool HUD::uploadConstData(Dar::UploadHandle uploadHandle) {
	auto app = Dar::getApp();

	HUDConstData data = {};
	data.projection = Mat4{
		Vec4{ 2.f,  0.f, 0.f, 0.f},
		Vec4{ 0.f, -2.f, 0.f, 0.f},
		Vec4{ 0.f,  0.f, 1.f, 0.f},
		Vec4{-1.f,  1.f, 0.f, 1.f},
	};
	data.width = float(app->getWidth());
	data.height = float(app->getHeight());

	/// Initialize the MVP constant buffer resource if needed
	const int frameIndex = hudRenderer.getBackbufferIndex();
	auto &resManager = Dar::getResourceManager();
	return resManager.uploadBufferData(uploadHandle, constData[frameIndex], reinterpret_cast<void*>(&data), sizeof(HUDConstData));
}

bool HUD::uploadWidgetData(const Vector<WidgetData>& data, Dar::UploadHandle uploadHandle) {
	if (widgetDataBuffer.getSize() < data.size() * sizeof(WidgetData)) {
		widgetDataBuffer.init(sizeof(WidgetData), data.size());
	}

	return widgetDataBuffer.upload(uploadHandle, reinterpret_cast<const void*>(data.data()));
}
