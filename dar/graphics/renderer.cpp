#include "renderer.h"

#include "d3d12/includes.h"
#include "framework/app.h"
#include "utils/defines.h"

#include "reslib/resource_library.h"

#include "imgui.h"
#include "imgui/backends/imgui_impl_dx12.h"
#include "imgui/backends/imgui_impl_glfw.h"

#include <filesystem>

namespace Dar {

Renderer::Renderer() : device(nullptr), framePipeline(nullptr) {}

bool Renderer::init(Device& d, bool renderToScr) {
	device = &d;
	
	renderToScreen = renderToScr;

	if (renderToScreen) {
		backbufferIndex = device->getBackbuffer().getCurrentBackBufferIndex();
		device->getBackbuffer().registerRenderer(this);
	} else {
		// As this renderer is only auxiliary
		// don't use the current backbuffer index
		backbufferIndex = 0;
	}

	allowTearing = device->getAllowTearing();

	return true;
}

void Renderer::deinit() {
}

void Renderer::beginFrame() {
	auto& backbuffer = device->getBackbuffer();

	if (renderToScreen) {
		backbufferIndex = backbuffer.getCurrentBackBufferIndex();
	} else {
		backbufferIndex = (backbufferIndex + 1) % Dar::FRAME_COUNT;
	}

	waitFrameResources(backbufferIndex);

	++numRenderedFrames;
}

void Renderer::endFrame() {
	if (renderToScreen) {
		auto& backbuffer = device->getBackbuffer();

		const UINT syncInterval = settings.vSyncEnabled ? 1 : 0;
		const UINT presentFlags = allowTearing && !settings.vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;

		RETURN_ON_ERROR(backbuffer.present(syncInterval, presentFlags), , "Failed to execute command list!");
	}
}

FenceValue Renderer::renderFrame(const FrameData &frameData) {
	if (renderToScreen) {
		getApp()->setBackbufferRendered();
	}

	if (frameData.renderCommands.empty()) { // nothing to do!
		return 0;
	}

	auto &cmdQueue = device->getCommandQueue();

	cmdQueue.addCommandListForExecution(populateCommandList(frameData));

	auto &resman = getResourceManager();
	uploadsToWait[backbufferIndex] = frameData.uploadsToWait;
	for (auto uploadCtxHandle : uploadsToWait[backbufferIndex]) {
		resman.gpuWaitUpload(device->getCommandQueue(), uploadCtxHandle);
	}

	for (auto fence : frameData.fencesToWait) {
		device->getCommandQueue().gpuWaitForFenceValue(fence);
	}

	FenceValue fenceValue = cmdQueue.executeCommandLists();
	fenceValues[backbufferIndex] = fenceValue;

	return fenceValue;
}

void Renderer::waitFrameResources(int backbufferIdx) {
	auto &cmdQueue = device->getCommandQueue();
	cmdQueue.cpuWaitForFenceValue(fenceValues[backbufferIdx]);

	auto &resman = getResourceManager();
	for (auto handle : uploadsToWait[backbufferIdx]) {
		resman.cpuWaitUpload(handle);
	}

	uploadsToWait[backbufferIdx].clear();
}

void Renderer::onBackbufferResize() {
	if (renderToScreen) {
		backbufferIndex = device->getBackbuffer().getCurrentBackBufferIndex();
	}
}

void Renderer::setFramePipeline(FramePipeline *fp) {
	framePipeline = fp;
}

void Renderer::renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle) {
	if (!settings.useImGui) {
		return;
	}

	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	getApp()->drawUI();

	ImGui::Render();

	cmdList.setRenderTargets(&rtvHandle, nullptr, 1);
	cmdList.setDescriptorHeap(device->getImGuiSRVHeap().GetAddressOf());

	ComPtr<ID3D12GraphicsCommandList> gCmdList;
	cmdList.getComPtr().As(&gCmdList);
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), gCmdList.Get());
}

CommandList Renderer::populateCommandList(const FrameData &frameData) {
	CommandList cmdList = device->getCommandList();
	if (framePipeline == nullptr) {
		LOG_FMT(Warning, "Empty frame pipeline!");
		return cmdList;
	}

	auto app = getApp();

	// Cache the rtv handle. The last rtv handle will be used for the ImGui rendering.
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = {};
	for (int renderPassIndex = 0; renderPassIndex < framePipeline->getNumPasses(); ++renderPassIndex) {
		if (frameData.isPassEmpty(renderPassIndex)) {
			continue;
		}

		RenderPass &renderPass = framePipeline->getPass(renderPassIndex);
		renderPass.begin(cmdList, backbufferIndex);

		auto &shaderResources = frameData.shaderResources[renderPassIndex];
		const SizeType numShaderResources = shaderResources.size();
		auto &srvHeap = renderPass.srvHeap[backbufferIndex];
		if (numShaderResources > 0 && (!srvHeap || srvHeap.getNumViews() < numShaderResources)) {
			srvHeap.init(
				device->getDevicePtr(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
				static_cast<UINT>(numShaderResources),
				true
			);
		}

		srvHeap.reset();
		for (auto &res : shaderResources) {
			ResourceHandle handle = {};
			bool isTex = false;
			switch (res.type) {
			case FrameData::ShaderResourceType::Data:
				handle = res.data->getHandle();
				srvHeap.addBufferSRV(handle.get(), static_cast<UINT>(res.data->getNumElements()), static_cast<UINT>(res.data->getElementSize()));
				break;
			case FrameData::ShaderResourceType::Texture:
				handle = res.tex->getHandle();
				srvHeap.addTexture2DSRV(*res.tex);
				break;
			case FrameData::ShaderResourceType::TextureCube:
				handle = res.tex->getHandle();
				srvHeap.addTextureCubeSRV(*res.tex);
				break;
			default:
				dassert(false);
				break;
			}
			cmdList.transition(handle, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			if (isTex) {
			} else {
			}
		}

		if (renderPass.srvHeap[backbufferIndex]) {
			cmdList.setDescriptorHeap(renderPass.srvHeap[backbufferIndex].getAddressOf());
		}

		auto viewport = renderPass.viewport;
		if (viewport.Width <= 0) {
			viewport.Width = float(app->getWidth());
		}
		if (viewport.Height <= 0) {
			viewport.Height = float(app->getHeight());
		}
		cmdList.setViewport(viewport);
		cmdList.setScissorRect(scissorRect);

		cmdList.setRootSignature(renderPass.pipeline.getRootSignature());

		for (auto &cb : frameData.constantBuffers) {
			cmdList.transition(cb.handle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
			cmdList.setConstantBufferView(cb.rootParameterIndex, cb.handle);
		}

		const bool hasDepthBuffer = renderPass.depthBufferAttachment.valid();
		const int numRenderTargets = static_cast<int>(renderPass.renderTargetDescs.size());
		rtvHandle = renderPass.rtvHeap[backbufferIndex].getCPUHandle(0);
		const D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = hasDepthBuffer ? renderPass.depthBufferAttachment.getCPUHandle() : D3D12_CPU_DESCRIPTOR_HANDLE{ 0 };

		for (int i = 0; i < numRenderTargets; ++i) {
			cmdList.clearRenderTarget(renderPass.rtvHeap[backbufferIndex].getCPUHandle(i));
		}

		if (hasDepthBuffer && renderPass.depthBufferAttachment.clearDepthBuffer()) {
			cmdList.transition(renderPass.depthBufferAttachment.getResourceHandle(0), D3D12_RESOURCE_STATE_DEPTH_WRITE);
			cmdList.clearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH);
		}

		cmdList.setPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		if (frameData.vertexBuffer) {
			cmdList.transition(frameData.vertexBuffer->bufferHandle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
			cmdList.setVertexBuffers(&frameData.vertexBuffer->bufferView, 1);
		}
		if (frameData.indexBuffer) {
			cmdList.transition(frameData.indexBuffer->bufferHandle, D3D12_RESOURCE_STATE_INDEX_BUFFER);
			cmdList.setIndexBuffer(&frameData.indexBuffer->bufferView);
		}

		cmdList.setRenderTargets(
			numRenderTargets > 0 ? &rtvHandle : nullptr,
			hasDepthBuffer ? &dsvHandle : nullptr,
			numRenderTargets
		);

		frameData.renderCommands[renderPassIndex].execCommands(cmdList);

		renderPass.end(cmdList);
	}

	renderUI(cmdList, rtvHandle);
	cmdList.transition(device->getBackbuffer().getHandle(backbufferIndex), D3D12_RESOURCE_STATE_PRESENT);

	return cmdList;
}

} // namespace Dar
