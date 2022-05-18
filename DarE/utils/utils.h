#pragma once

#include "d3d12/includes.h"
#include "utils/defines.h"

namespace Dar {

WString getCommandQueueNameByType(D3D12_COMMAND_LIST_TYPE type);
WString getCommandListNameByType(D3D12_COMMAND_LIST_TYPE type);

} // namespace Dar