#include "render_pass.h"
#include "frame_data.h"
#include "framework/app.h"

namespace Dar {

void RenderPassDesc::attach(const RenderPassAttachment &rpa) {
	attachments.push_back(rpa);
}

void RenderPassDesc::setViewport(int width, int height) {
	auto app = Dar::getApp();

	if (width > 0) {
		viewportWidth = float(width);
	} else {
		viewportWidth = float(app->getWidth());
	}

	if (height > 0) {
		viewportHeight = float(height);
	} else {
		viewportHeight = float(app->getHeight());
	}
}

void RenderPass::init(ComPtr<ID3D12Device> device, Backbuffer *backbuf, const RenderPassDesc &rpd) {
	auto &psoDesc = rpd.psoDesc;
	pipeline.init(device, psoDesc);
	
	viewport = D3D12_VIEWPORT{ 0.f, 0.f, rpd.viewportWidth, rpd.viewportHeight, 0.f, 1.f };

	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	D3D12_RENDER_PASS_RENDER_TARGET_DESC rtDesc = {};
	bool hasDepthStencilBuffer = false;
	bool hasBackbufferAttachment = false;
	for (int i = 0; i < rpd.attachments.size(); ++i) {
		auto &attachment = rpd.attachments[i];
		switch (attachment.getType()) {
		case RenderPassAttachmentType::RenderTarget:
			if (attachment.isBackbuffer()) {
				dassert(!hasBackbufferAttachment);
				dassert(backbuf);
				if (!backbuf) {
					break;
				}

				backbuffer = backbuf;
				hasBackbufferAttachment = true;
			}

			// TODO: Add support for different begining/ending accesses if needed
			rtDesc = {};
			rtDesc.cpuDescriptor = { NULL }; // We will update that during rendering
			rtDesc.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
			rtDesc.BeginningAccess.Clear.ClearValue.Format = attachment.isBackbuffer() ? backbuffer->getFormat() : attachment.getFormat();
			rtDesc.BeginningAccess.Clear.ClearValue.Color[0] = 0.f;
			rtDesc.BeginningAccess.Clear.ClearValue.Color[1] = 0.f;
			rtDesc.BeginningAccess.Clear.ClearValue.Color[2] = 0.f;
			rtDesc.BeginningAccess.Clear.ClearValue.Color[3] = 0.f;
			rtDesc.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;

			renderTargetDescs.push_back(rtDesc);
			renderTargetAttachments.push_back(attachment);

			break;
		case RenderPassAttachmentType::DepthStencil:
			if (hasDepthStencilBuffer) {
				dassert(false);
				break;
			}

			depthStencilDesc.cpuDescriptor = attachment.getCPUHandle();
			if (rpd.attachments[i].clearDepthBuffer()) {
				depthStencilDesc.DepthBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
				depthStencilDesc.DepthBeginningAccess.Clear.ClearValue.Format = attachment.getFormat();
				depthStencilDesc.DepthBeginningAccess.Clear.ClearValue.DepthStencil = { 1.f, 0 };
			} else {
				depthStencilDesc.DepthBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE;
			}
			depthStencilDesc.DepthEndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
			depthBufferAttachment = attachment;

			hasDepthStencilBuffer = true;
			break;
		}
	}

	dassert(psoDesc.numRenderTargets == renderTargetAttachments.size());
	dassert((psoDesc.depthStencilBufferFormat != DXGI_FORMAT_UNKNOWN) == hasDepthStencilBuffer);

	for (int i = 0; i < FRAME_COUNT; ++i) {
		rtvHeap[i].init(
			device.Get(),
			D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
			static_cast<int>(renderTargetAttachments.size()), /*numDescriptors*/
			true /*shaderVisible*/
		);
	}
}

void RenderPass::begin(CommandList &cmdList, int backbufferIndex) {
	cmdList.setPipelineState(pipeline.getPipelineState());

	// Prepare the rtv heap. Do this each frame in order not to deal with
	// tracking when the RTV texture is changed and stuff like that.
	rtvHeap[backbufferIndex].reset();
	bool hasBackBuffer = false;
	for (int i = 0; i < renderTargetAttachments.size(); ++i) {
		ID3D12Resource *rtRes = nullptr;
		const bool isBackBuffer = renderTargetAttachments[i].isBackbuffer();
		if (isBackBuffer) {
			dassert(backbuffer && !hasBackBuffer); // Only one backbuffer attachment is allowed!
			if (backbuffer != nullptr) {
				rtRes = backbuffer->getBufferResource(backbufferIndex);
			}
			hasBackBuffer = true;
		} else {
			rtRes = renderTargetAttachments[i].getBufferResource(backbufferIndex);
		}

		dassert(rtRes != nullptr);

		rtvHeap[backbufferIndex].addRTV(rtRes, nullptr);
		renderTargetDescs[i].cpuDescriptor = rtvHeap[backbufferIndex].getCPUHandle(i);
		if (isBackBuffer && backbuffer) {
			cmdList.transition(backbuffer->getHandle(backbufferIndex), D3D12_RESOURCE_STATE_RENDER_TARGET);
		} else {
			cmdList.transition(
				renderTargetAttachments[i].getResourceHandle(backbufferIndex),
				D3D12_RESOURCE_STATE_RENDER_TARGET
			);
		}
	}

	/*bool hasDepthStencil = depthBufferAttachment.valid();
	cmdList->BeginRenderPass(static_cast<UINT>(renderTargetDescs.size()), renderTargetDescs.data(), hasDepthStencil ? &depthStencilDesc : NULL, D3D12_RENDER_PASS_FLAG_NONE);*/
}

void RenderPass::end(CommandList &) {
	/*cmdList->EndRenderPass();*/
}

void RenderPass::deinit() {
	pipeline.deinit();
	for (int i = 0; i < FRAME_COUNT; ++i) {
		rtvHeap[i].deinit();
		srvHeap[i].deinit();
	}
}

} // namespace Dar
