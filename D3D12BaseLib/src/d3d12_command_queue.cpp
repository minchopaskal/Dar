#include "d3d12_command_queue.h"

#include "d3d12_defines.h"
#include "d3d12_res_tracker.h"
#include "d3d12_utils.h"

CommandQueue::CommandQueue(D3D12_COMMAND_LIST_TYPE type) :
	device(nullptr),
	fenceValue(0),
	fenceEvent(nullptr),
	type(type) { }

void CommandQueue::init(ComPtr<ID3D12Device8> device) {
	this->device = device;
	
	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	cqDesc.Type = type;
	RETURN_ON_ERROR(
		device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&commandQueue)), ,
		"Failed to create command queue!"
	);

	commandQueue->SetName(getCommandQueueNameByType(type).c_str());

	RETURN_ON_ERROR(
		device->CreateFence(fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), ,
		"Failed to create a fence!"
	);

	fenceEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	RETURN_ON_ERROR(
		static_cast<HRESULT>(fenceEvent != nullptr), ,
		"Failed to create fence event!"
	);
}

CommandList CommandQueue::getCommandList() {
	// TODO: try lock and if cannot lock just create new allocator+cmdList
	auto lock = commandListsPoolCS.lock();

	CommandList cmdList;
	ComPtr<ID3D12CommandAllocator> cmdAllocator;

	if (!commandAllocatorsPool.empty() && fenceCompleted(commandAllocatorsPool.front().fenceValue)) {
		cmdAllocator = commandAllocatorsPool.front().cmdAllocator;
		commandAllocatorsPool.pop();
	} else {
		cmdAllocator = createCommandAllocator();
	}

	RETURN_ON_ERROR(
		cmdAllocator->Reset(), cmdList,
		"Failed to reset command allocator!"
	);

	if (!commandListsPool.empty()) {
		cmdList = commandListsPool.front();
		commandListsPool.pop();
	} else {
		cmdList = createCommandList(cmdAllocator);
	}

	RETURN_ON_ERROR(
		cmdList->Reset(cmdAllocator.Get(), nullptr), cmdList,
		"Failed to reset the command list!"
	);

	RETURN_ON_ERROR(
		cmdList->SetPrivateDataInterface(__uuidof(ID3D12CommandAllocator), cmdAllocator.Get()), cmdList,
		"Failure CommandList::SetPrivateDataInterface"
	);

	return cmdList;
}

void CommandQueue::addCommandListForExecution(CommandList &&commandList) {
	if (!commandList.isValid()) {
		OutputDebugString("Invalid command list submitted!\n");
		return;
	}

	auto lock = pendingCommandListsCS.lock();
	pendingCommandListsQueue.push_back(commandList);
}

UINT64 CommandQueue::executeCommandLists() {
	Vector<ID3D12CommandList*> cmdListsToExecute;
	Vector<ID3D12CommandAllocator*> cmdAllocators;
	
	Vector<CommandList> auxCommandLists;

	UINT64 fenceVal;
	// Lock the pending command lists queue until the command lists and any needed resource barriers are exctracted
	{
		auto lock = pendingCommandListsCS.lock();
		cmdListsToExecute.reserve(pendingCommandListsQueue.size() * 4); // reserve more space in case cmd lists have pending barriers
		cmdAllocators.reserve(pendingCommandListsQueue.size() * 4);

		auxCommandLists.reserve(pendingCommandListsQueue.size() * 3);

		// For each pending command list, check its pending resource barriers
		for (int i = 0; i < pendingCommandListsQueue.size(); ++i) {
			CommandList &cmdList = pendingCommandListsQueue[i];

			Vector<PendingResourceBarrier> &pendingBarriers = cmdList.getPendingResourceBarriers();
			CommandList cmdListPendingBarriers;
			Vector<CD3DX12_RESOURCE_BARRIER> resBarriers;

			// For each pending barrier check the (sub)resource's state and only push the barrier if it's needed
			for (int j = 0; j < pendingBarriers.size(); ++j) {
				PendingResourceBarrier &b = pendingBarriers[j];

				if (b.subresourceIndex == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) { // we need to transition all of the subresources
					Vector<D3D12_RESOURCE_STATES> states;
					ResourceTracker::getLastGlobalState(b.res, states);

					for (int k = 0; k < states.size(); ++k) {
						if (states[k] == b.stateAfter) {
							continue;;
						}
						resBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
							b.res,
							states[k],
							b.stateAfter,
							k
						));

					}
				} else { // Only one of the subresources needs transitioning
					D3D12_RESOURCE_STATES state;
					ResourceTracker::getLastGlobalStateForSubres(b.res, state, b.subresourceIndex);
					if (state != b.stateAfter) {
						resBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
							b.res,
							state,
							b.stateAfter,
							b.subresourceIndex
						));
					}
				}
			}
			pendingBarriers.clear();

			// We have resource barriers we need to call before executing the command list. Initialize the auxiliary command list and call the barriers from there.
			if (resBarriers.size() > 0) {
				cmdListPendingBarriers = getCommandList();
				cmdListPendingBarriers->ResourceBarrier((UINT)resBarriers.size(), resBarriers.data());
			}

			// Save command allocator of main and auxiliary cmd list to be later pushed into the allocators pool when the fence value is known
			ID3D12CommandAllocator *cmdAllocator;
			UINT dataSize = sizeof(cmdAllocator);
			RETURN_FALSE_ON_ERROR(
				cmdList->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &cmdAllocator),
				"Failure CommandList::GetPrivateData"
			);
			cmdAllocators.push_back(cmdAllocator);

			// If we had any resource barriers to call, we save the command list and its allocator in the respective pools
			if (cmdListPendingBarriers.isValid()) {
				RETURN_FALSE_ON_ERROR(
					cmdListPendingBarriers->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &cmdAllocator),
					"Failure CommandList::GetPrivateData"
				);
				cmdAllocators.push_back(cmdAllocator);

				cmdListPendingBarriers->Close();

				auxCommandLists.push_back(cmdListPendingBarriers);
				cmdListsToExecute.push_back(cmdListPendingBarriers.get());
			}

			// Set global state of resources to what the last states in the command list were
			// so on the next command list in the pendingCommandListsQueue would know how to deal
			// with its pending barriers.
			cmdList.resolveLastStates();

			cmdList->Close();
			cmdListsToExecute.push_back(cmdList.get());
		}

		commandQueue->ExecuteCommandLists((UINT)cmdListsToExecute.size(), cmdListsToExecute.data());
		fenceVal = signal();

		for (int i = 0; i < pendingCommandListsQueue.size(); ++i) {
			commandListsPool.emplace(pendingCommandListsQueue[i]);
		}
		pendingCommandListsQueue.clear();
	}

	for (int i = 0; i < auxCommandLists.size(); ++i) {
		commandListsPool.emplace(auxCommandLists[i]);
	}

	for (int i = 0; i < cmdAllocators.size(); ++i) {
		commandAllocatorsPool.emplace(CommandAllocator{ cmdAllocators[i], fenceVal });
		cmdAllocators[i]->Release();
	}

	return fenceVal;
}

ComPtr<ID3D12CommandQueue> CommandQueue::getCommandQueue() const {
	return commandQueue;
}

UINT64 CommandQueue::signal() {
	UINT64 fenceVal = ++fenceValue;
	RETURN_ON_ERROR(commandQueue->Signal(fence.Get(), fenceVal), 0, "Failed to signal command queue!");

	return fenceVal;
}

bool CommandQueue::fenceCompleted(UINT64 fenceVal) const {
	return fence->GetCompletedValue() >= fenceVal;
}

#include <unordered_set>

void CommandQueue::waitForFenceValue(UINT64 fenceVal) {
	while (!fenceCompleted(fenceVal)) {
		RETURN_ON_ERROR(
			fence->SetEventOnCompletion(fenceVal, fenceEvent), ,
			"Failed to set fence event on completion!"
		);
		WaitForSingleObject(fenceEvent, INFINITE);
	}
}

void CommandQueue::flush() {
	UINT64 fenceVal = signal();
	waitForFenceValue(fenceVal);
}

ComPtr<ID3D12CommandAllocator> CommandQueue::createCommandAllocator() {
	ComPtr<ID3D12CommandAllocator> cmdAllocator;
	RETURN_NULL_ON_ERROR(
		device->CreateCommandAllocator(type, IID_PPV_ARGS(&cmdAllocator)),
		"Failed to create command allocator!"
	);

	return cmdAllocator;
}

CommandList CommandQueue::createCommandList(const ComPtr<ID3D12CommandAllocator> &cmdAllocator) {
	CommandList cmdList;
	cmdList.init(device, cmdAllocator, type);

	return cmdList;
}
