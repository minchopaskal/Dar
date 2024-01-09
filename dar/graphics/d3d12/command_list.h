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

	bool init(const ComPtr<ID3D12Device> &device, D3D12_COMMAND_LIST_TYPE type);

	void transition(ResourceHandle resource, D3D12_RESOURCE_STATES stateAfter, const UINT subresourceIndex = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
	void resolveLastStates();

	Vector<PendingResourceBarrier> &getPendingResourceBarriers();

	ID3D12GraphicsCommandList *get() {
		return cmdList.Get();
	}

	ComPtr<ID3D12CommandList> getComPtr() {
		return cmdList;
	}

	const ComPtr<ID3D12CommandList> getComPtr() const {
		return cmdList;
	}

	ID3D12CommandList** getAddressOf() {
		return reinterpret_cast<ID3D12CommandList**>(cmdList.GetAddressOf());
	}

	void dispatch(uint32_t threadGroupCount);
	void setRootSignature(ID3D12RootSignature *rootSignature, bool compute);
	void setConstantBufferView(unsigned int rootParameterIndex, ResourceHandle constBufferHandle, bool compute);
	void drawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertex, uint32_t startInstance);
	void drawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndex, uint32_t baseVertex, uint32_t startInstance);
	void resourceBarriers(const Vector<D3D12_RESOURCE_BARRIER> &barriers);
	void copyResource(ResourceHandle dest, ResourceHandle src);
	void copyBufferRegion(ResourceHandle dest, ResourceHandle src, SizeType size);
	void setRenderTargets(const D3D12_CPU_DESCRIPTOR_HANDLE *rtvHandle, const D3D12_CPU_DESCRIPTOR_HANDLE *dsvHandle, uint32_t numRenderTargets);
	void setDescriptorHeap(ID3D12DescriptorHeap *const *heap);
	void setViewport(const D3D12_VIEWPORT &viewport);
	void setScissorRect(const D3D12_RECT &rect);
	void clearRenderTarget(D3D12_CPU_DESCRIPTOR_HANDLE handle);
	void clearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE handle, D3D12_CLEAR_FLAGS flags);
	void setPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY type);
	void setVertexBuffers(D3D12_VERTEX_BUFFER_VIEW *views, uint32_t count);
	void setIndexBuffer(D3D12_INDEX_BUFFER_VIEW *view);
	void setPipelineState(ID3D12PipelineState *state);
	D3D12Result reset(ID3D12CommandAllocator *cmdAllocator);
	D3D12Result close();
	D3D12Result setPrivateData(REFGUID guid, const IUnknown *data);
	D3D12Result getPrivateData(REFGUID guid, uint32_t *pDataSize, void *pData);

private:
	void resourceBarriersImpl(const Vector<D3D12_RESOURCE_BARRIER> &barriers);
	void flushCurrentPendingBarriers();

private:
	using SubresStates = Vector<D3D12_RESOURCE_STATES>;
	using LastStates = Map<SizeType, SubresStates>;

	ComPtr<ID3D12GraphicsCommandList4> cmdList;
	Vector<PendingResourceBarrier> initialPendingBarriers;
	Vector<D3D12_RESOURCE_BARRIER> currentPendingBarriers;
	LastStates lastStates;
	D3D12_COMMAND_LIST_TYPE type;
	bool valid;
};

} // namespace Dar