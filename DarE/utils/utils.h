#pragma once

#include "d3d12/includes.h"
#include "utils/defines.h"

#include "cppcoro/include/cppcoro/generator.hpp"

namespace Dar {

WString getCommandQueueNameByType(D3D12_COMMAND_LIST_TYPE type);
WString getCommandListNameByType(D3D12_COMMAND_LIST_TYPE type);
int getPixelSizeFromFormat(DXGI_FORMAT format);
DXGI_FORMAT getDepthFormatAsNormal(DXGI_FORMAT);

cppcoro::generator<String> generateFileLines(const String &filename);

} // namespace Dar