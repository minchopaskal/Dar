#pragma once

#include "render_command_list.h"

namespace Dar {

RenderCommandList::~RenderCommandList() {
	delete[] memory;
	memory = nullptr;
	size = 0;
}

void RenderCommandList::execCommands(CommandList &cmdList) const {
	using RenderCommandIterator = Byte*;
	RenderCommandIterator it = memory;
	SizeType rcSize = 0;

	auto perCmdCallback = [&cmdList, &it, &rcSize]<RenderCommandConcept T>(T*) -> void {
		reinterpret_cast<T*>(it)->exec(cmdList);
		rcSize = sizeof(T);
	};

	while (it != memory + size) {
		auto *renderCommand = std::bit_cast<RenderCommandInvalid*>(it);
		switch (renderCommand->type) {
		using enum RenderCommandType;
		case DrawInstanced:
			perCmdCallback((RenderCommandDrawInstanced*)nullptr);
			break;
		case DrawIndexedInstanced:
			perCmdCallback((RenderCommandDrawIndexedInstanced*)nullptr);
			break;
		case SetConstantBuffer:
			perCmdCallback((RenderCommandSetConstantBuffer*)nullptr);
			break;
		case Transition:
			perCmdCallback((RenderCommandSetConstantBuffer*)nullptr);
			break;
		default:
			dassert(false);
			break;
		}
		it += rcSize;
	}
}

} // namespace Dar
