#include "d3d12_command_queue.h"

#include "d3d12_defines.h"
#include "d3d12_utils.h"

CommandQueue::CommandQueue(D3D12_COMMAND_LIST_TYPE type) :
	device(nullptr),
	fenceValue(0),
	fenceEvent(nullptr),
	type(type) { }

void CommandQueue::init(ComPtr<ID3D12Device2> device) {
	this->device = device;

	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	cqDesc.Type = type;
	RETURN_ON_ERROR(
		device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&commandQueue)), ,
		"Failed to create command queue!\n"
	);

	commandQueue->SetName(getCommandQueueNameByType(type).c_str());

	RETURN_ON_ERROR(
		device->CreateFence(fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), ,
		"Failed to create a fence!\n"
	);

	fenceEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	RETURN_ON_ERROR(
		static_cast<HRESULT>(fenceEvent != nullptr), ,
		"Failed to create fence event!\n"
	);
}

ComPtr<ID3D12GraphicsCommandList2> CommandQueue::getCommandList() {
	ComPtr<ID3D12GraphicsCommandList2> cmdList;
	ComPtr<ID3D12CommandAllocator> cmdAllocator;

	if (!commandAllocatorQueue.empty() && fenceCompleted(commandAllocatorQueue.front().fenceValue)) {
		cmdAllocator = commandAllocatorQueue.front().cmdAllocator;
		commandAllocatorQueue.pop();
	} else {
		cmdAllocator = createCommandAllocator();
	}

	RETURN_FALSE_ON_ERROR(
		cmdAllocator->Reset(),
		"Failed to reset command allocator!\n"
	);

	if (!commandListQueue.empty()) {
		cmdList = commandListQueue.front();
		commandListQueue.pop();
	} else {
		cmdList = createCommandList(cmdAllocator);
	}

	RETURN_FALSE_ON_ERROR(
		cmdList->Reset(cmdAllocator.Get(), nullptr),
		"Failed to reset the command list!\n"
	);

	RETURN_FALSE_ON_ERROR(
		cmdList->SetPrivateDataInterface(__uuidof(ID3D12CommandAllocator), cmdAllocator.Get()),
		"Failure CommandList::SetPrivateDataInterface"
	);

	return cmdList;
}

UINT64 CommandQueue::executeCommandList(ComPtr<ID3D12GraphicsCommandList2> cmdList) {
	cmdList->Close();

	ID3D12CommandAllocator *cmdAllocator;
	UINT dataSize = sizeof(cmdAllocator);
	RETURN_FALSE_ON_ERROR(
		cmdList->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &cmdAllocator),
		"Failure CommandList::GetPrivateData"
	);

	ID3D12CommandList *const commandLists[] = { cmdList.Get() };
	commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	UINT64 fenceVal = signal();

	commandAllocatorQueue.emplace(CommandAllocator{ cmdAllocator, fenceVal });
	commandListQueue.push(cmdList);

	cmdAllocator->Release();
		
	return fenceVal;
}

ComPtr<ID3D12CommandQueue> CommandQueue::getCommandQueue() const {
	return commandQueue;
}

UINT64 CommandQueue::signal() {
	UINT64 fenceVal = ++fenceValue;
	RETURN_ON_ERROR(commandQueue->Signal(fence.Get(), fenceVal), 0, "Failed to signal command queue!\n");

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
			"Failed to set fence event on completion!\n"
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
		"Failed to create command allocator!\n"
	);

	return cmdAllocator;
}

ComPtr<ID3D12GraphicsCommandList2> CommandQueue::createCommandList(ComPtr<ID3D12CommandAllocator> cmdAllocator) {
	ComPtr<ID3D12GraphicsCommandList2> cmdList;

	RETURN_FALSE_ON_ERROR(
		device->CreateCommandList(0, type, cmdAllocator.Get(), nullptr, IID_PPV_ARGS(&cmdList)),
		"Failed to create command list!\n"
	);

	cmdList->SetName(getCommandListNameByType(type).c_str());

	RETURN_FALSE_ON_ERROR(
		cmdList->Close(),
		"Failed to close the command list after creation!\n"
	);

	return cmdList;
}
