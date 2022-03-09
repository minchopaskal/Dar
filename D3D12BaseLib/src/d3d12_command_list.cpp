#include "d3d12_command_list.h"

#include "d3d12_resource_manager.h"
#include "d3d12_utils.h"

#include "d3dx12.h"

CommandList::CommandList() : valid(false), type(D3D12_COMMAND_LIST_TYPE::D3D12_COMMAND_LIST_TYPE_DIRECT) { }

bool CommandList::isValid() const {
	return valid;
}

bool CommandList::init(const ComPtr<ID3D12Device8> &device, const ComPtr<ID3D12CommandAllocator> &cmdAllocator, D3D12_COMMAND_LIST_TYPE type) {
	this->type = type;

	cmdList.Reset();

	RETURN_FALSE_ON_ERROR(
		device->CreateCommandList1(0, type, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmdList)),
		"Failed to create command list!"
	);

	cmdList->SetName(getCommandListNameByType(type).c_str());

	valid = true;

	return true;
}

void CommandList::transition(ResourceHandle resource, D3D12_RESOURCE_STATES stateAfter, const UINT subresourceIndex) {
	static const D3D12_RESOURCE_STATES UNKNOWN_STATE = static_cast<D3D12_RESOURCE_STATES>(0xffffffff);

	if (!valid) {
		return;
	}

	ResourceManager &resManager = getResourceManager();

	ID3D12Resource *res = resManager.getID3D12Resource(resource);
	auto pushPendingBarrier = [&](D3D12_RESOURCE_STATES &state) {
		PendingResourceBarrier barrier;
		barrier.resHandle = resource;
		barrier.stateAfter = stateAfter;
		barrier.subresourceIndex = subresourceIndex;
		pendingBarriers.push_back(barrier);
	};

	auto pushPendingBarrierIf = [&](D3D12_RESOURCE_STATES &state, const D3D12_RESOURCE_STATES &cond) -> bool {
		if (state == cond) {
			pushPendingBarrier(state);

			state = stateAfter;
			return true;
		}
		return false;
	};
	
	if (lastStates.find(resource) != lastStates.end()) { // the resource was already transitioned once
		SubresStates &states = lastStates[resource];
		for (int i = 0; i < states.size(); ++i) {	// Check whether the subresource was transitioned
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
			cmdList->ResourceBarrier(1, &resBarrier);

			states[i] = stateAfter;
		}

	} else {
		// this is the first time we encounter the subresource for the current recording.
		// Instead of transitioning here, we will add the barrier to the list of pending
		// barriers because we don't know the 'beforeState' of the resource, yet
		lastStates[resource].resize(resManager.getSubresourcesCount(resource));
		SubresStates &states = lastStates[resource];
		for (int i = 0; i < states.size(); ++i) {
			if (subresourceIndex == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
				states[i] = stateAfter;
			} else if (i == subresourceIndex) {
				states[subresourceIndex] = stateAfter;
			} else {
				states[i] = UNKNOWN_STATE;
			}

			pushPendingBarrier(states[i]);
		}
	}
}

void CommandList::setMVPBuffer(ResourceHandle mvpBufferHandle) {
	cmdList->SetGraphicsRootConstantBufferView(0, mvpBufferHandle->GetGPUVirtualAddress());
}

void CommandList::setConstantBufferView(unsigned int index, ResourceHandle constBufferHandle) {
	dassert(index > 0);
	cmdList->SetGraphicsRootConstantBufferView(index, constBufferHandle->GetGPUVirtualAddress());
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

Vector<PendingResourceBarrier>& CommandList::getPendingResourceBarriers() {
	return pendingBarriers;
}
