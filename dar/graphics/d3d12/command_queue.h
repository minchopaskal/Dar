#pragma once

#include <D3d12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include "async/async.h"
#include "d3d12/command_list.h"
#include "utils/defines.h"

namespace Dar {

struct CommandQueue {
	CommandQueue(D3D12_COMMAND_LIST_TYPE type);

	void init(ComPtr<ID3D12Device> device);
	void deinit();

	CommandList getCommandList();
	void addCommandListForExecution(CommandList &&commandList);

	/// Execute all command lists previously submitted via
	/// addCommandListsForExecution.
	/// Before excuting them any pending resource barriers recorded via
	/// calls to CommandList::transition() are resolved. This is done in order to automate
	/// the handling of the resources' states. \see CommandList::transition() which only wants to know
	/// the after state - the before state is handled by the app.
	/// NOTE! Any call to addCommandListForExecution will block until the currently
	/// pending command lists are executed.
	FenceValue executeCommandLists();

	ComPtr<ID3D12CommandQueue> getCommandQueue() const;

	ComPtr<ID3D12Fence> getFence() const;

	FenceValue signal();
	bool fenceCompleted(FenceValue fenceVal) const;
	void waitForFenceValue(FenceValue fenceVal);
	void flush();

	void waitQueueForFenceValue(const CommandQueue &queue, FenceValue val);

private:
	ComPtr<ID3D12CommandAllocator> createCommandAllocator();
	CommandList createCommandList();

private:
	struct CommandAllocator {
		ComPtr<ID3D12CommandAllocator> cmdAllocator;
		FenceValue fenceValue;
	};

	ComPtr<ID3D12Device> device;
	ComPtr<ID3D12CommandQueue> commandQueue;

	Vector<CommandList> pendingCommandListsQueue;
	Queue<CommandList> commandListsPool;
	Queue<CommandAllocator> commandAllocatorsPool;

	CriticalSection commandListsPoolCS;
	CriticalSection pendingCommandListsCS;

	ComPtr<ID3D12Fence> fence;
	HANDLE fenceEvent;
	FenceValue fenceValue;

	D3D12_COMMAND_LIST_TYPE type;
};

} // namespace Dar