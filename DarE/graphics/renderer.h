#pragma once

#include "d3d12/depth_buffer.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/vertex_index_buffer.h"
#include "d3d12/texture_res.h"
#include "math/dar_math.h"
#include "utils/defines.h"

#include <concepts>

namespace Dar {

constexpr UINT FRAME_COUNT = 2;

struct RenderTarget {
	RenderTarget() : numFramesInFlight(0) {}
	
	void init(TextureInitData &texInitData, UINT numFramesInFlight);

	ResourceHandle getHandleForFrame(UINT frameIndex) const;

	ID3D12Resource* getBufferResourceForFrame(UINT frameIndex) const;

	/// Resize the render target buffers with the given dimensions.
	/// Note: Expects the render target to not be in use!
	/// @param width The new width of the render target.
	/// @param height The new height of the render target.
	void resizeRenderTarget(int width, int height);

	void setName(const WString &name);

	DXGI_FORMAT getFormat() {
		if (numFramesInFlight == 0) {
			return DXGI_FORMAT_UNKNOWN;
		}

		return rtTextures[0].getFormat();
	}

	int getWidth() const {
		if (numFramesInFlight == 0) {
			return DXGI_FORMAT_UNKNOWN;
		}

		return rtTextures[0].getWidth();
	}

	int getHeight() const {
		if (numFramesInFlight == 0) {
			return DXGI_FORMAT_UNKNOWN;
		}

		return rtTextures[0].getHeight();
	}

private:
	TextureResource rtTextures[FRAME_COUNT];
	UINT numFramesInFlight;
};


struct Backbuffer {
	bool init(ComPtr<IDXGIFactory4> dxgiFactory, CommandQueue &commandQueue, bool allowTearing);

	ID3D12Resource *getBufferResource(UINT backbufferIndex) const {
		return backBuffers[backbufferIndex].Get();
	}

	ResourceHandle getHandle(UINT backbufferIndex) const {
		return backBuffersHandles[backbufferIndex];
	}

	bool registerInResourceManager();

	bool resize();

	UINT getCurrentBackBufferIndex() const;

	DXGI_FORMAT getFormat() const;

	HRESULT present(UINT syncInterval, UINT flags) const;

public:
	// Swap chain resources
	ComPtr<IDXGISwapChain4> swapChain; ///< Pointer to the swap chain
	ComPtr<ID3D12Resource> backBuffers[FRAME_COUNT]; ///< The RT resources
	ResourceHandle backBuffersHandles[FRAME_COUNT]; ///< Handles to the RT resources in the resource manager
};


struct ConstantBuffer {
	ResourceHandle bufferHandle;
	int rootParameterIndex;
};

struct FrameData {
	Vector<ConstantBuffer> constantBuffers;
	VertexBuffer *vertexBuffer;
	IndexBuffer *indexBuffer;

	void clear() {
		vertexBuffer = nullptr;
		indexBuffer = nullptr;
		constantBuffers.clear();
	}
};

enum class RenderPassAttachmentType {
	RenderTarget,
	DepthStencil,

	Invalid
};

struct RenderPassAttachment {
	RenderPassAttachment() : rt(nullptr), backbuffer(false), type(RenderPassAttachmentType::Invalid) {}

	static RenderPassAttachment renderTarget(RenderTarget *rt) {
		RenderPassAttachment res = { };
		res.rt = rt;
		res.type = RenderPassAttachmentType::RenderTarget;

		return res;
	}

	static RenderPassAttachment renderTargetBackbuffer() {
		RenderPassAttachment res = { };
		res.rt = nullptr;
		res.backbuffer = true;
		res.type = RenderPassAttachmentType::RenderTarget;

		return res;
	}

	static RenderPassAttachment depthStencil(DepthBuffer *db, bool clear) {
		RenderPassAttachment res = { };
		res.depthBuffer = db;
		res.clear = clear;
		res.type = RenderPassAttachmentType::DepthStencil;
		
		return res;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getCPUHandle() const {
		switch (type) {
		case RenderPassAttachmentType::DepthStencil:
			return depthBuffer->getCPUHandle();
			// We don't know the cpu descriptor of the render target
			// since the TextureResource doesn't know about the heap it resides in.
		case RenderPassAttachmentType::RenderTarget:
		default:
			return { NULL };
		}
	}

	ID3D12Resource *getBufferResource(UINT frameIndex) {
		switch (type) {
		case RenderPassAttachmentType::RenderTarget:
			return rt->getBufferResourceForFrame(frameIndex);
		case RenderPassAttachmentType::DepthStencil:
			return depthBuffer->getBufferResource();
		default:
			return nullptr;
		}
	}

	ResourceHandle getResourceHandle(UINT frameIndex) {
		switch (type) {
		case RenderPassAttachmentType::RenderTarget:
			return rt->getHandleForFrame(frameIndex);
		case RenderPassAttachmentType::DepthStencil:
			return depthBuffer->getBufferHandle();
		default:
			return INVALID_RESOURCE_HANDLE;
		}
	}

	DXGI_FORMAT getFormat() const {
		switch (type) {
		case RenderPassAttachmentType::RenderTarget:
			return rt->getFormat();
		case RenderPassAttachmentType::DepthStencil:
			return depthBuffer->getFormatAsTexture();
		default:
			return DXGI_FORMAT_UNKNOWN;
		}
	}

	RenderPassAttachmentType getType() const {
		return type;
	}

	bool valid() const {
		return type != RenderPassAttachmentType::Invalid;
	}

	bool clearDepthBuffer() const {
		if (type != RenderPassAttachmentType::DepthStencil) {
			return false;
		}

		return clear;
	}

	bool isBackbuffer() const {
		return backbuffer;
	}

private:
	union {
		struct {
			RenderTarget *rt; ///< Attachment's underlying texture if it's type is RenderTarget. If null use the backbuffer as RT.
			bool backbuffer; ///< Flag indicating we are rendering to the backbuffer. Ignores rt.
		};
		struct {
			DepthBuffer *depthBuffer; ///< DepthBuffer
			bool clear; ///< Flag indicating the depth buffer should be cleared at the beginning of the rennder pass.
		};
	};
	RenderPassAttachmentType type; ///< Type of the attachment.
};


/// Resource initialization function that will be called at the beginning
/// of the render pass. Its idea is to transition resources and initialize
/// the render pass' SRV heap, but can be used in any other way.
using RenderPassResourceInitCallback = void (*)(const FrameData &frameData, CommandList &cmdList, DescriptorHeap &srvHeap, UINT backbufferIndex, void *args);

using DrawCallback = void (*)(CommandList &cmdList, void *args);
struct RenderPassDesc {
	Vector<RenderPassAttachment> attachments; 
	PipelineStateDesc psoDesc = {}; ///< Description of the pipeline state. The render pass will construct it.
	DrawCallback drawCb = nullptr;
	RenderPassResourceInitCallback setupCb = nullptr;
	void *args = nullptr; ///< Arguments passed to both setupCb and drawCb.

	/// Add a render pass attachment
	void attach(const RenderPassAttachment &rpa);
};

struct RenderSettings {
	int showGBuffer = 0;
	bool enableNormalMapping= 1;
	bool enableFXAA = 0;
	bool useImGui = 0; ///< Flag indicating whether ImGui will be used for drawing UI.
	bool vSyncEnabled = 0;
	bool spotLightON = 0;
};

struct RenderPass;

struct Renderer {
	Renderer();

	void addRenderPass(const RenderPassDesc &rpd);
	void compilePipeline();

	void init();
	void deinit();

	void flush();

	void beginFrame();
	void renderFrame(const FrameData &frameData);
	void endFrame();

	bool registerBackBuffersInResourceManager();
	bool resizeBackBuffers();

	RenderSettings& getSettings() {
		return settings;
	}

	ComPtr<ID3D12Device> getDevice() {
		return device;
	}

	UINT64 getNumRenderedFrames() const {
		return numRenderedFrames;
	}

	UINT getBackbufferIndex() const {
		return backbufferIndex;
	}

	CommandList getCommandList() {
		return commandQueueDirect.getCommandList();
	}

private:
	bool initDevice();
	bool initImGui();

	void renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle);

	CommandList populateCommandList(const FrameData &frameData);

private:
	ComPtr<ID3D12Device> device; ///< DX12 device used across all the classes

	/// General command queue. Primarily used for draw calls.
	/// Copy calls are handled by the resource manager
	CommandQueue commandQueueDirect;

	Backbuffer backbuffer; ///< Object holding the data for the backbuffers.

	// Imgui
	ComPtr<ID3D12DescriptorHeap> imguiSRVHeap; ///< SRVHeap used by Dear ImGui for font drawing

	UINT backbufferIndex = 0; ///< Current backbuffer index

	UINT64 numRenderedFrames = 0;

	Vector<RenderPassDesc> renderPassesDesc;
	Vector<RenderPass*> renderPasses;
	Byte *renderPassesStorage;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[FRAME_COUNT] = { 0, 0 };

	RenderSettings settings = {};

	// viewport
	D3D12_VIEWPORT viewport = { };
	D3D12_RECT scissorRect = { 0, 0, LONG_MAX, LONG_MAX };

	bool allowTearing = false;

	/// Flag that indicates ImGui was already shutdown. Since ImGui doesn't do check for double delete
	/// in its DX12 implementation of Shutdown, we have to do it manually.
	bool imGuiShutdown = false;
};

}