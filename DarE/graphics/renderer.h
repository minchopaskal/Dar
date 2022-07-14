#pragma once

#include "d3d12/data_buffer.h"
#include "d3d12/depth_buffer.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/vertex_index_buffer.h"
#include "d3d12/texture_res.h"
#include "math/dar_math.h"
#include "utils/defines.h"

#include <functional>

#include <concepts>


namespace Dar {

constexpr UINT FRAME_COUNT = 2;

struct RenderTarget {
	std::function<void()> f;
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

	const TextureResource& getTextureResource(int backbufferIndex) const {
		return rtTextures[backbufferIndex];
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

	void deinit();

public:
	// Swap chain resources
	ComPtr<IDXGISwapChain4> swapChain; ///< Pointer to the swap chain
	ComPtr<ID3D12Resource> backBuffers[FRAME_COUNT]; ///< The RT resources
	ResourceHandle backBuffersHandles[FRAME_COUNT]; ///< Handles to the RT resources in the resource manager
};

struct RenderCommand {
	static RenderCommand drawInstanced(
		UINT vertexCount,
		UINT instanceCount,
		UINT startVertex,
		UINT startInstance
	);

	static RenderCommand drawIndexedInstanced(
		UINT indexCount,
		UINT instanceCount,
		UINT startIndex,
		UINT baseVertex,
		UINT startInstance
	);

	static RenderCommand setConstantBuffer(
		ResourceHandle constBufferHandle,
		UINT rootIndex
	);

	static RenderCommand transition(
		ResourceHandle resource,
		D3D12_RESOURCE_STATES toState,
		UINT subresIndex = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
	);

	void exec(CommandList &cmdList) const;

private:
	RenderCommand() : type(RenderCommandType::Invalid) {}

	enum class RenderCommandType {
		DrawInstanced,
		DrawIndexedInstanced,
		SetConstantBuffer,
		Transition,

		Invalid,
		// ...
	} type;
	union {
		struct {
			UINT vertexCountDI;
			UINT instanceCountDI;
			UINT startVertexDI;
			UINT startInstanceDI;
		};

		struct {
			UINT indexCountDII;
			UINT instanceCountDII;
			UINT startIndexDII;
			UINT baseVertexDII;
			UINT startInstanceDII;
		};

		struct {
			ResourceHandle constBufferHandleSCB;
			UINT rootIndexSCB;
		};

		struct {
			ResourceHandle resourceT;
			D3D12_RESOURCE_STATES toStateT;
			UINT subresIndexT;
		};
	};
};

struct FrameData {
	friend struct Renderer;
	friend struct RenderPass;

	void beginFrame(const Renderer &renderer);

	void endFrame(const Renderer &renderer) {}

	void setVertexBuffer(VertexBuffer *vb) {
		vertexBuffer = vb;
	}

	void setIndexBuffer(IndexBuffer *ib) {
		indexBuffer = ib;
	}

	void addConstResource(const ResourceHandle &handle, int rootParamterIndex) {
		constantBuffers.emplace_back(handle, rootParamterIndex);
	}

	void addDataBufferResource(const DataBufferResource &dataBuf, int passIndex) {
		ShaderResource shaderResource = {};
		shaderResource.data = &dataBuf;
		shaderResource.type = ShaderResourceType::Data;
		shaderResources[passIndex].push_back(shaderResource);
	}

	void addTextureResource(const TextureResource &texRes, int passIndex) {
		ShaderResource shaderResource = {};
		shaderResource.tex = &texRes;
		shaderResource.type = ShaderResourceType::Texture;
		shaderResources[passIndex].emplace_back(shaderResource);
	}
	
	void addRenderCommand(RenderCommand &&renderCmd, int passIndex) {
		renderCommands[passIndex].emplace_back(std::move(renderCmd));
	}

private:
	struct ConstantResource {
		ResourceHandle handle;
		int rootParameterIndex;
	};

	enum class ShaderResourceType {
		Data,
		Texture
	};

	struct ShaderResource {
		union {
			const TextureResource *tex;
			const DataBufferResource *data;
		};
		ShaderResourceType type;
	};

	/// Constant buffers are bind to all passes of the pipeline.
	Vector<ConstantResource> constantBuffers;

	/// Shader resources for each pass of the pipeline.
	/// Mapping is 1-1, so for pass 0 the renderer would look
	/// its resources in shaderResources[0].
	Vector<Vector<ShaderResource>> shaderResources;

	/// List of render commands for each pass. The list is executed as is,
	/// i.e it preserves the order of the commands as they were passed.
	Vector<Vector<RenderCommand>> renderCommands;

	VertexBuffer *vertexBuffer = nullptr;
	IndexBuffer *indexBuffer = nullptr;
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
using RenderPassResourceInitCallback = std::function<void(const FrameData &frameData, CommandList &cmdList, DescriptorHeap &srvHeap, UINT backbufferIndex)>;

using DrawCallback = void (*)(CommandList &cmdList, void *args);
struct RenderPassDesc {
	friend struct Renderer;
	friend struct RenderPass;

	void setPipelineStateDesc(const PipelineStateDesc &psoDesc) {
		this->psoDesc = psoDesc;
	}

	/// Add a render pass attachment
	void attach(const RenderPassAttachment &rpa);

private:
	Vector<RenderPassAttachment> attachments;
	PipelineStateDesc psoDesc = {}; ///< Description of the pipeline state. The render pass will construct it.
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

	SizeType getNumPasses() const {
		return renderPasses.size();
	}

private:
	bool initDevice();
	bool initImGui();
	bool deinitImGui();

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