#pragma once

#include "backbuffer.h"
#include "render_target.h"
#include "d3d12/depth_buffer.h"
#include "d3d12/pipeline_state.h"

namespace Dar {

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

using DrawCallback = void (*)(CommandList &cmdList, void *args);
struct RenderPassDesc {
	friend class Renderer;
	friend struct RenderPass;

	void setPipelineStateDesc(const PipelineStateDesc &pd) {
		psoDesc = pd;
	}

	/// Add a render pass attachment
	void attach(const RenderPassAttachment &rpa);

	/// @brief Set viewport dimensions.
	/// @default Negative values mean using the App's width and height.
	void setViewport(int width, int height);

private:
	Vector<RenderPassAttachment> attachments;
	PipelineStateDesc psoDesc = {}; ///< Description of the pipeline state. The render pass will construct it.
	float viewportWidth = -1;
	float viewportHeight = -1;
};

struct FrameData;
struct RenderPass {
	/// Initialize the render pass given the description.
	/// Creates the pipeline state (TODO: load PSO from cache)
	/// @param device Device used for the render pass initialization steps.
	/// @param rpd Render pass description.
	/// @param frameCount How many frames are rendered at the same time. Used for determining the size of the SRV and RTV heaps.
	void init(ComPtr<ID3D12Device> device, Backbuffer *backbuffer, const RenderPassDesc &rpd);

	void begin(CommandList &cmdList, int backbufferIndex);
	
	void end(CommandList &cmdList);
	
	void deinit();

public:
	PipelineState pipeline;
	Vector<RenderPassAttachment> renderTargetAttachments;
	RenderPassAttachment depthBufferAttachment;
	Vector<D3D12_RENDER_PASS_RENDER_TARGET_DESC> renderTargetDescs;
	Backbuffer *backbuffer;
	DescriptorHeap rtvHeap[FRAME_COUNT];
	DescriptorHeap srvHeap[FRAME_COUNT];
	D3D12_RENDER_PASS_DEPTH_STENCIL_DESC depthStencilDesc;
	D3D12_VIEWPORT viewport;
};

} // namespace Dar
