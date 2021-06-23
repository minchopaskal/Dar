#pragma once

#include "d3d12_includes.h"
#include "d3d12_defines.h"

struct PendingResourceBarrier {
	ID3D12Resource *res;
	D3D12_RESOURCE_STATES stateAfter;
	UINT subresourceIndex;
};

struct CommandList {
	CommandList();

	bool isValid() const;

	bool init(const ComPtr<ID3D12Device8> &device, const ComPtr<ID3D12CommandAllocator> &cmdAllocator, D3D12_COMMAND_LIST_TYPE type);

	void transition(ComPtr<ID3D12Resource> &resource, D3D12_RESOURCE_STATES stateAfter, const UINT subresourceIndex = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
	void resolveLastStates();

	Vector<PendingResourceBarrier>& getPendingResourceBarriers();

	// TODO: wrap command list operations so we don't need this potential bomb.
	ID3D12GraphicsCommandList2* operator->() {
		return cmdList.Get();
	}

	ID3D12GraphicsCommandList2* get() {
		return cmdList.Get();
	}

	ComPtr<ID3D12GraphicsCommandList2> getComPtr() {
		return cmdList;
	}

private:
	using SubresStates = Vector<D3D12_RESOURCE_STATES>;
	using LastStates = Map<ID3D12Resource*, SubresStates>;

	ComPtr<ID3D12GraphicsCommandList2> cmdList;
	Vector<PendingResourceBarrier> pendingBarriers;
	LastStates lastStates;
	D3D12_COMMAND_LIST_TYPE type;
	bool valid;
};