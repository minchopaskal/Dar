#pragma once

#include <D3d12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include "d3d12_async.h"
#include "d3d12_command_list.h"
#include "d3d12_defines.h"

struct CommandQueue {
	CommandQueue(D3D12_COMMAND_LIST_TYPE type);

	void init(ComPtr<ID3D12Device8> device);

	CommandList getCommandList();
	void addCommandListForExecution(CommandList &&commandList);

	UINT64 executeCommandLists();

	ComPtr<ID3D12CommandQueue> getCommandQueue() const;

	ComPtr<ID3D12Fence> getFence() const;

	UINT64 signal();
	bool fenceCompleted(UINT64 fenceVal) const;
	void waitForFenceValue(UINT64 fenceVal);
	void flush();

private:
	ComPtr<ID3D12CommandAllocator> createCommandAllocator();
	CommandList createCommandList(const ComPtr<ID3D12CommandAllocator> &cmdAllocator);

private:
	struct CommandAllocator {
		ComPtr<ID3D12CommandAllocator> cmdAllocator;
		UINT64 fenceValue;
	};

	ComPtr<ID3D12Device8> device;
	ComPtr<ID3D12CommandQueue> commandQueue;

	Vector<CommandList> pendingCommandListsQueue;
	Queue<CommandList> commandListsPool;
	Queue<CommandAllocator> commandAllocatorsPool;

	CriticalSection commandListsPoolCS;
	CriticalSection pendingCommandListsCS;

	ComPtr<ID3D12Fence> fence;
	HANDLE fenceEvent;
	UINT64 fenceValue;

	D3D12_COMMAND_LIST_TYPE type;
};