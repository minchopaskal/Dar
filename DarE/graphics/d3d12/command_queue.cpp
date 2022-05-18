#include "d3d12/command_queue.h"

#include "d3d12/resource_manager.h"
#include "utils/defines.h"
#include "utils/utils.h"

#include "d3dx12.h"

namespace Dar {

CommandQueue::CommandQueue(D3D12_COMMAND_LIST_TYPE type) :
	device(nullptr),
	fenceValue(0),
	fenceEvent(nullptr),
	type(type) {}

void CommandQueue::init(ComPtr<ID3D12Device> device) {
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

	// We only check the front of the queue for available command allocators.
	// We make sure that all the command allocators pushed to the queue after it have bigger
	// fence values, thus if the fence ensuring the first command allocator is free to use is not ready yet
	// no allocator after it is ready either.
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
	pendingCommandListsQueue.push_back(std::move(commandList));
}

UINT64 CommandQueue::executeCommandLists() {
	ResourceManager &resManager = getResourceManager();

	// The command lists in the pending command lists queue will go here
	// to be executed together.
	Vector<ID3D12CommandList*> pendingCmdListsToExecute;

	// Each command list's allocator we will execute in this method
	// will be cached at the end of the function for reuse in future command lists.
	Vector<ID3D12CommandAllocator*> cmdAllocators;

	// Array of the resource barriers we want to execute before submitting
	// the pending command lists.
	Vector<D3D12_RESOURCE_BARRIER> resBarriers;

	UINT64 fenceVal;

	// Lock the pending command lists queue until the command lists and any needed resource barriers are exctracted
	{
		auto lock = pendingCommandListsCS.lock();
		pendingCmdListsToExecute.reserve(pendingCommandListsQueue.size());
		cmdAllocators.reserve(pendingCommandListsQueue.size() + 1); // Reserve space for the allocators we want to cache. +1 for the pendingBarriersCmdList allocator 

		// For each pending command list, check its pending resource barriers
		for (int i = 0; i < pendingCommandListsQueue.size(); ++i) {
			CommandList &cmdList = pendingCommandListsQueue[i];

			Vector<PendingResourceBarrier> &pendingBarriers = cmdList.getPendingResourceBarriers();

			// For each pending barrier check the (sub)resource's state and only push the barrier if it's needed
			for (int j = 0; j < pendingBarriers.size(); ++j) {
				PendingResourceBarrier &b = pendingBarriers[j];
				ID3D12Resource *res = resManager.getID3D12Resource(b.resHandle);

				if (b.subresourceIndex == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) { // we need to transition all of the subresources
					Vector<D3D12_RESOURCE_STATES> states;
					resManager.getLastGlobalState(b.resHandle, states);

					for (int k = 1; k < states.size(); ++k) {
						dassert(states[k] == states[0]);
					}

					if (states[0] != b.stateAfter) {
						resBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
							res,
							states[0],
							b.stateAfter,
							D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
						));
					}
				} else { // Only one of the subresources needs transitioning
					D3D12_RESOURCE_STATES state;
					resManager.getLastGlobalStateForSubres(b.resHandle, state, b.subresourceIndex);
					if (state != b.stateAfter) {
						resBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
							res,
							state,
							b.stateAfter,
							b.subresourceIndex
						));
					}
				}
			}
			pendingBarriers.clear();

			// Save command allocator of main and auxiliary cmd list to be later pushed into the allocators pool when the fence value is known
			ID3D12CommandAllocator *cmdAllocator;
			UINT dataSize = sizeof(cmdAllocator);
			RETURN_FALSE_ON_ERROR(
				cmdList->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &cmdAllocator),
				"Failure CommandList::GetPrivateData"
			);
			cmdAllocators.push_back(cmdAllocator);

			// Set global state of the resources to what the last states in the frame command list were
			// so when the next frame's command list is recorded the pendingCommandListsQueue would know how to deal
			// with its pending barriers.
			cmdList.resolveLastStates();

			cmdList->Close();
			pendingCmdListsToExecute.push_back(cmdList.get());
		}

		// We have resource barriers we need to call before executing the command list. Initialize the auxiliary command list and call the barriers from there.
		CommandList cmdListPendingBarriers;
		if (resBarriers.size() > 0) {
			cmdListPendingBarriers = getCommandList();
			cmdListPendingBarriers->ResourceBarrier((UINT)resBarriers.size(), resBarriers.data());
			cmdListPendingBarriers->Close();

			// If we had any resource barriers to call, we save the command list and its allocator in the respective pools
			ID3D12CommandAllocator *cmdAllocator;
			UINT dataSize = sizeof(cmdAllocator);
			RETURN_FALSE_ON_ERROR(
				cmdListPendingBarriers->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &cmdAllocator),
				"Failure CommandList::GetPrivateData"
			);
			cmdAllocators.push_back(cmdAllocator);
		}

		// Execute the pending barriers command lists first to ensure the proper states
		// were resolved for the resources. Executing all of the command lists together may result
		// in driver optimizations leading to reordering of the ResourceBarrier commands.
		if (cmdListPendingBarriers.isValid()) {
			commandQueue->ExecuteCommandLists(1, cmdListPendingBarriers.getAddressOf());
		}

		if (pendingCmdListsToExecute.size()) {
			commandQueue->ExecuteCommandLists((UINT)pendingCmdListsToExecute.size(), pendingCmdListsToExecute.data());
		}
		fenceVal = signal();

		if (cmdListPendingBarriers.isValid()) {
			commandListsPool.emplace(cmdListPendingBarriers);
		}
		for (int i = 0; i < pendingCommandListsQueue.size(); ++i) {
			commandListsPool.emplace(pendingCommandListsQueue[i]);
		}
		pendingCommandListsQueue.clear();
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

ComPtr<ID3D12Fence> CommandQueue::getFence() const {
	return fence;
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
	if (commandQueue == nullptr) {
		return;
	}
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

} // namespace Dar