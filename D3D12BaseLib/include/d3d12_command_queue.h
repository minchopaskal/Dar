#pragma once

#include <D3d12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include "d3d12_defines.h"

struct CommandQueue {
	CommandQueue(D3D12_COMMAND_LIST_TYPE type);

	void init(ComPtr<ID3D12Device2> device);

	ComPtr<ID3D12GraphicsCommandList2> getCommandList();
	UINT64 executeCommandList(ComPtr<ID3D12GraphicsCommandList2> commandList);

	ComPtr<ID3D12CommandQueue> getCommandQueue() const;

	UINT64 signal();
	bool fenceCompleted(UINT64 fenceVal) const;
	void waitForFenceValue(UINT64 fenceVal);
	void flush();

private:
	ComPtr<ID3D12CommandAllocator> createCommandAllocator();
	ComPtr<ID3D12GraphicsCommandList2> createCommandList(ComPtr<ID3D12CommandAllocator> cmdAllocator);

private:
	struct CommandAllocator {
		ComPtr<ID3D12CommandAllocator> cmdAllocator;
		UINT64 fenceValue;
	};

	ComPtr<ID3D12Device2> device;
	ComPtr<ID3D12CommandQueue> commandQueue;

	Queue<ComPtr<ID3D12GraphicsCommandList2>> commandListQueue;
	Queue<CommandAllocator> commandAllocatorQueue;

	ComPtr<ID3D12Fence> fence;
	HANDLE fenceEvent;
	UINT64 fenceValue;

	D3D12_COMMAND_LIST_TYPE type;
};