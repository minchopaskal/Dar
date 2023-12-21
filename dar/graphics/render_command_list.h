#pragma once

#include "core.h"
#include "d3d12/command_list.h"
#include "utils/defines.h"

namespace Dar {

enum class RenderCommandType {
	DrawInstanced,
	DrawIndexedInstanced,
	SetConstantBuffer,
	Transition,

	Invalid,
};

template <class T>
concept RenderCommandConcept = requires (const T &x, CommandList &l) {
	x.exec(l); // Existence of non-static void T::exec(CommandList&)
	T::type; // Existence of member variable T::type
	std::is_same_v<decltype(T::type), RenderCommandType>; // type of T::type == RenderCommandType1
};

struct RenderCommandInvalid {
	RenderCommandType type = RenderCommandType::Invalid;

	void exec(CommandList&) { }
};

struct RenderCommandDrawInstanced {
	RenderCommandType type = RenderCommandType::DrawInstanced;

	RenderCommandDrawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertex, uint32_t startInstance)
		: vc(vertexCount), ic(instanceCount), sv(startVertex), si(startInstance) { }

	void exec(CommandList &cmdList) const {
		cmdList.drawInstanced(vc, ic, sv, si);
	}

private:
	uint32_t vc, ic, sv, si;
};

struct RenderCommandDrawIndexedInstanced {
	RenderCommandType type = RenderCommandType::DrawIndexedInstanced;

	RenderCommandDrawIndexedInstanced(
		uint32_t indexCount,
		uint32_t instanceCount,
		uint32_t startIndex,
		uint32_t baseVertex,
		uint32_t startInstance
	) : idxCnt(indexCount), ic(instanceCount), si(startIndex), bv(baseVertex), sInst(startInstance) {}

	void exec(CommandList &cmdList) const {
		cmdList.drawIndexedInstanced(idxCnt, ic, si, bv, sInst);
	}

private:
	uint32_t idxCnt, ic, si, bv, sInst;
};

struct RenderCommandSetConstantBuffer {
	RenderCommandType type = RenderCommandType::SetConstantBuffer;

	RenderCommandSetConstantBuffer(ResourceHandle constBufferHandle, uint32_t rootIndex) 
		: constBufferHandle(constBufferHandle), rootIndex(rootIndex) {}

	void exec(CommandList &cmdList) const {
		cmdList.transition(constBufferHandle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		cmdList.setConstantBufferView(rootIndex, constBufferHandle);
	}

private:
	ResourceHandle constBufferHandle;
	uint32_t rootIndex;
};

struct RenderCommandTransition {
	RenderCommandType type = RenderCommandType::Transition;

	RenderCommandTransition(
		ResourceHandle resource,
		D3D12_RESOURCE_STATES toState,
		uint32_t subresIndex = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
	) 
		: resource(resource), toState(toState), subresIndex(subresIndex) {}

	void exec(CommandList &cmdList) const {
		cmdList.transition(resource, toState, subresIndex);
	}

private:
	ResourceHandle resource;
	D3D12_RESOURCE_STATES toState;
	uint32_t subresIndex;
};

// TODO: when execCommands is called cache the commands in a bundle for next frame use.
struct RenderCommandList {
	RenderCommandList() : memory(nullptr), size(0) {}
	
	~RenderCommandList();

	void addRenderCommand(RenderCommandConcept auto rc) {
		SizeType rcSize = sizeof(rc);

		// TODO: This would be done best with a custom allocator(per frame one in this exact case).
		Byte *newMemory = new Byte[size + rcSize];
		memcpy(newMemory, memory, size);
		memcpy(newMemory + size, &rc, rcSize);
		delete[] memory;
		memory = newMemory;
		size += rcSize;
	}

	void execCommands(CommandList &cmdList) const;

	bool empty() const {
		return memory == nullptr || size == 0;
	}

private:
	Byte *memory;
	SizeType size;
};

} // namespace Dar
