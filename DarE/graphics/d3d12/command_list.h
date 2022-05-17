#pragma once

#include "d3d12/includes.h"
#include "d3d12/resource_handle.h"
#include "utils/defines.h"

namespace Dar {

struct PendingResourceBarrier {
	ResourceHandle resHandle;
	D3D12_RESOURCE_STATES stateAfter;
	UINT subresourceIndex;
};

struct CommandList {
	CommandList();

	bool isValid() const;

	bool init(const ComPtr<ID3D12Device> &device, const ComPtr<ID3D12CommandAllocator> &cmdAllocator, D3D12_COMMAND_LIST_TYPE type);

	void transition(ResourceHandle resource, D3D12_RESOURCE_STATES stateAfter, const UINT subresourceIndex = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
	void setConstantBufferView(unsigned int rootParameterIndex, ResourceHandle constBufferHandle);
	void resolveLastStates();

	Vector<PendingResourceBarrier> &getPendingResourceBarriers();

	// TODO: wrap command list operations so we don't need this potential bomb.
	ID3D12GraphicsCommandList4 *operator->() {
		return cmdList.Get();
	}

	ID3D12GraphicsCommandList *get() {
		return cmdList.Get();
	}

	ComPtr<ID3D12CommandList> getComPtr() {
		return cmdList;
	}

	const ComPtr<ID3D12CommandList> getComPtr() const {
		return cmdList;
	}

	ID3D12CommandList **getAddressOf() {
		return reinterpret_cast<ID3D12CommandList **>(cmdList.GetAddressOf());
	}

private:
	using SubresStates = Vector<D3D12_RESOURCE_STATES>;
	using LastStates = Map<SizeType, SubresStates>;

	ComPtr<ID3D12GraphicsCommandList4> cmdList;
	Vector<PendingResourceBarrier> pendingBarriers;
	LastStates lastStates;
	D3D12_COMMAND_LIST_TYPE type;
	bool valid;
};

} // namespace Dar