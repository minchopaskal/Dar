#include "d3d12_command_list.h"

#include "d3d12_defines.h"
#include "d3d12_res_tracker.h"
#include "d3d12_utils.h"

CommandList::CommandList() : valid(false), type(D3D12_COMMAND_LIST_TYPE_DIRECT) { }

bool CommandList::isValid() const {
	return valid;
}

bool CommandList::init(const ComPtr<ID3D12Device8> &device, const ComPtr<ID3D12CommandAllocator> &cmdAllocator, D3D12_COMMAND_LIST_TYPE type) {
	this->type = type;

	RETURN_FALSE_ON_ERROR(
		device->CreateCommandList1(0, type, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmdList)),
		"Failed to create command list!"
	);

	cmdList->SetName(getCommandListNameByType(type).c_str());

	valid = true;

	return true;
}

void CommandList::transition(ComPtr<ID3D12Resource>& resource, D3D12_RESOURCE_STATES stateAfter, const UINT subresourceIndex) {
	static const UINT UNKNOWN_STATE = 0xffffffff;

	if (!valid) {
		return;
	}

	ID3D12Resource *res = resource.Get();
	auto pushPendingBarrier = [&](D3D12_RESOURCE_STATES &state) -> bool {
		if (static_cast< UINT >(state) == UNKNOWN_STATE) {
			PendingResourceBarrier barrier;
			barrier.res = res;
			barrier.stateAfter = stateAfter;
			barrier.subresourceIndex = subresourceIndex;
			pendingBarriers.push_back(barrier);

			state = stateAfter;
			return true;
		}
		return false;
	};
	
	if (lastStates.find(res) != lastStates.end()) { // the resource was already transitioned once
		SubresStates &states = lastStates[res];
		for (int i = 0; i < states.size(); ++i) {		// Check whether the subresource was transitioned
			if (subresourceIndex == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
				// the subresource was never touched in this command list until now
				pushPendingBarrier(states[i]);
			} else if (subresourceIndex != i) {
				continue;
			}

			// subresourceIndex == i
			if (pushPendingBarrier(states[i])) {
				continue;
			}

			D3D12_RESOURCE_BARRIER resBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
				res,
				states[i],
				stateAfter,
				subresourceIndex
			);
			cmdList->ResourceBarrier(1, &resBarrier);
		}

	} else { // this is the first time we encounter the subresource.
		// Instead of transitioning here, we will add the barrier to the list of pending
		// barriers because we don't know the 'beforeState' of the resource, yet
		lastStates[res].resize(ResourceTracker::getSubresourcesCount(res));
		SubresStates states = lastStates[res];
		for (int i = 0; i < states.size(); ++i) {
			if (subresourceIndex == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
				states[i] = stateAfter;
			} else if (i == subresourceIndex) {
				states[subresourceIndex] = stateAfter;
			} else {
				states[i] = static_cast<D3D12_RESOURCE_STATES>(UNKNOWN_STATE);
			}

			pushPendingBarrier(states[i]);
		}
	}
}

void CommandList::resolveLastStates() {
	for (auto it = lastStates.begin(); it != lastStates.end(); ++it) {
		for (int i = 0; i < it->second.size(); ++i) {
			ResourceTracker::setGlobalStateForSubres(it->first, it->second[i], i);
		}
	}
}

Vector<PendingResourceBarrier>& CommandList::getPendingResourceBarriers() {
	return pendingBarriers;
}
