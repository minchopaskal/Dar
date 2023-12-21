#include "d3d12/command_list.h"

#include "d3d12/resource_manager.h"
#include "utils/utils.h"

#include "d3dx12.h"

namespace Dar {

CommandList::CommandList() : valid(false), type(D3D12_COMMAND_LIST_TYPE::D3D12_COMMAND_LIST_TYPE_DIRECT) {}

bool CommandList::isValid() const {
	return valid;
}

bool CommandList::init(const ComPtr<ID3D12Device> &device, D3D12_COMMAND_LIST_TYPE t) {
	type = t;

	cmdList.Reset();

	ComPtr<ID3D12Device4> device4;
	RETURN_FALSE_ON_ERROR(device.As(&device4), "Failed to aquire ID3D12Device4 interface from device!");

	RETURN_FALSE_ON_ERROR(
		device4->CreateCommandList1(0, type, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmdList)),
		"Failed to create command list!"
	);

	cmdList->SetName(getCommandListNameByType(type).c_str());

	valid = true;

	return true;
}

void CommandList::resolveLastStates() {
	ResourceManager &resManager = getResourceManager();
	for (auto it = lastStates.begin(); it != lastStates.end(); ++it) {
		for (int i = 0; i < it->second.size(); ++i) {
			resManager.setGlobalStateForSubres(it->first, it->second[i], i);
		}
	}

	lastStates.clear();
}

Vector<PendingResourceBarrier> &CommandList::getPendingResourceBarriers() {
	return initialPendingBarriers;
}

void CommandList::transition(ResourceHandle resource, D3D12_RESOURCE_STATES stateAfter, const UINT subresourceIndex) {
	static const D3D12_RESOURCE_STATES UNKNOWN_STATE = static_cast<D3D12_RESOURCE_STATES>(0xffffffff);

	if (!valid) {
		return;
	}

	ResourceManager &resManager = getResourceManager();

	ID3D12Resource *res = resManager.getID3D12Resource(resource);
	auto pushPendingBarrier = [&]() {
		PendingResourceBarrier barrier;
		barrier.resHandle = resource;
		barrier.stateAfter = stateAfter;
		barrier.subresourceIndex = subresourceIndex;
		initialPendingBarriers.push_back(barrier);
	};

	auto pushPendingBarrierIf = [&](D3D12_RESOURCE_STATES &state, const D3D12_RESOURCE_STATES &cond) -> bool {
		if (state == cond) {
			pushPendingBarrier();

			state = stateAfter;
			return true;
		}
		return false;
	};

	if (lastStates.find(resource) != lastStates.end()) { // the resource was already transitioned once
		SubresStates &states = lastStates[resource];
		for (SizeType i = 0; i < states.size(); ++i) {	// Check whether the subresource was transitioned
			if (subresourceIndex != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES && subresourceIndex != i) {
				continue;
			}

			if (states[i] == stateAfter) {
				continue;
			}

			if (pushPendingBarrierIf(states[i], UNKNOWN_STATE)) {
				continue;
			}

			D3D12_RESOURCE_BARRIER resBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
				res,
				states[i],
				stateAfter,
				subresourceIndex
			);
			//cmdList->ResourceBarrier(1, &resBarrier);
			currentPendingBarriers.push_back(resBarrier);

			states[i] = stateAfter;
		}

	} else {
		// this is the first time we encounter the subresource for the current recording.
		// Instead of transitioning here, we will add the barrier to the list of pending
		// barriers because we don't know the 'beforeState' of the resource, yet
		lastStates[resource].resize(resManager.getSubresourcesCount(resource));

		SubresStates &states = lastStates[resource];
		for (SizeType i = 0; i < states.size(); ++i) {
			if (subresourceIndex == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
				states[i] = stateAfter;
				continue;
			}
			
			if (i == subresourceIndex) {
				states[subresourceIndex] = stateAfter;
			} else {
				states[i] = UNKNOWN_STATE;
			}
		}
		pushPendingBarrier();
	}
}

void CommandList::setConstantBufferView(unsigned int index, ResourceHandle constBufferHandle) {
	flushCurrentPendingBarriers();

	dassert(index >= 0);
	cmdList->SetGraphicsRootConstantBufferView(index, constBufferHandle->GetGPUVirtualAddress());
}

void CommandList::drawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertex, uint32_t startInstance) {
	flushCurrentPendingBarriers();
	get()->DrawInstanced(vertexCount, instanceCount, startVertex, startInstance);
}

void CommandList::drawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndex, uint32_t baseVertex, uint32_t startInstance) {
	flushCurrentPendingBarriers();
	get()->DrawIndexedInstanced(indexCount, instanceCount, startIndex, baseVertex, startInstance);
}

D3D12Result CommandList::reset(ID3D12CommandAllocator *cmdAllocator) {
	flushCurrentPendingBarriers();
	return get()->Reset(cmdAllocator, nullptr);
}

D3D12Result CommandList::close() {
	flushCurrentPendingBarriers();
	return get()->Close();
}

D3D12Result CommandList::setPrivateData(REFGUID guid, const IUnknown *data) {
	flushCurrentPendingBarriers();
	return get()->SetPrivateDataInterface(guid, data);
}

D3D12Result CommandList::getPrivateData(REFGUID guid, uint32_t *pDataSize, void *pData) {
	flushCurrentPendingBarriers();
	return get()->GetPrivateData(guid, pDataSize, pData);
}

void CommandList::resourceBarriers(const Vector<D3D12_RESOURCE_BARRIER> &barriers) {
	flushCurrentPendingBarriers();

	resourceBarriersImpl(barriers);
}

void CommandList::copyBufferRegion(ResourceHandle dest, ResourceHandle src, SizeType size) {
	flushCurrentPendingBarriers();
	auto &resManager = getResourceManager();

	auto destRes = resManager.getID3D12Resource(dest);
	auto srcRes = resManager.getID3D12Resource(src);

	get()->CopyBufferRegion(destRes, 0, srcRes, 0, size);
}

void CommandList::setRenderTargets(const D3D12_CPU_DESCRIPTOR_HANDLE *rtvHandle, const D3D12_CPU_DESCRIPTOR_HANDLE *dsvHandle, uint32_t numRenderTargets) {
	flushCurrentPendingBarriers();
	get()->OMSetRenderTargets(numRenderTargets, rtvHandle, true, dsvHandle);
}

void CommandList::setDescriptorHeap(ID3D12DescriptorHeap *const *heap) {
	flushCurrentPendingBarriers();
	get()->SetDescriptorHeaps(1, heap);
}

void CommandList::setViewport(const D3D12_VIEWPORT &viewport) {
	flushCurrentPendingBarriers();
	get()->RSSetViewports(1, &viewport);
}

void CommandList::setScissorRect(const D3D12_RECT &rect) {
	flushCurrentPendingBarriers();
	get()->RSSetScissorRects(1, &rect);
}

void CommandList::setRootSignature(ID3D12RootSignature *rootSignature) {
	flushCurrentPendingBarriers();
	get()->SetGraphicsRootSignature(rootSignature);
}

void CommandList::clearRenderTarget(D3D12_CPU_DESCRIPTOR_HANDLE handle) {
	flushCurrentPendingBarriers();
	static FLOAT clearColor[4] = { 0.f, 0.f, 0.f, 0.f };
	get()->ClearRenderTargetView(handle, clearColor, 0, nullptr);
}

void CommandList::clearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE handle, D3D12_CLEAR_FLAGS flags) {
	flushCurrentPendingBarriers();
	get()->ClearDepthStencilView(handle, flags, 1.f, 0, 0, nullptr);
}

void CommandList::setPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY topology) {
	flushCurrentPendingBarriers();
	get()->IASetPrimitiveTopology(topology);
}

void CommandList::setVertexBuffers(D3D12_VERTEX_BUFFER_VIEW *views, uint32_t count) {
	flushCurrentPendingBarriers();
	get()->IASetVertexBuffers(0, count, views);
}

void CommandList::setIndexBuffer(D3D12_INDEX_BUFFER_VIEW *view) {
	flushCurrentPendingBarriers();
	get()->IASetIndexBuffer(view);
}

void CommandList::setPipelineState(ID3D12PipelineState *state) {
	flushCurrentPendingBarriers();
	get()->SetPipelineState(state);
}

void CommandList::resourceBarriersImpl(const Vector<D3D12_RESOURCE_BARRIER> &barriers) {
	if (barriers.empty()) {
		return;
	}
	return get()->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
}

void CommandList::flushCurrentPendingBarriers() {
	resourceBarriersImpl(currentPendingBarriers);
	currentPendingBarriers.clear();
}

} // namespace Dar