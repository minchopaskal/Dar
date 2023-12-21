#include "utils/utils.h"

#include <codecvt>
#include <fstream>
#include <locale>

namespace Dar {

WString getPrefixedNameByType(D3D12_COMMAND_LIST_TYPE type, const wchar_t *prefix) {
	WString prefixStr{ prefix };
	switch (type) {
	case D3D12_COMMAND_LIST_TYPE_DIRECT:
		prefixStr.append(L"Direct");
		break;
	case D3D12_COMMAND_LIST_TYPE_COPY:
		prefixStr.append(L"Copy");
		break;
	case D3D12_COMMAND_LIST_TYPE_COMPUTE:
		prefixStr.append(L"Compute");
		break;
	default:
		prefixStr.append(L"Generic");
		break;
	}

	return prefixStr;
}

WString getCommandQueueNameByType(D3D12_COMMAND_LIST_TYPE type) {
	return getPrefixedNameByType(type, L"CommandQueue");
}

WString getCommandListNameByType(D3D12_COMMAND_LIST_TYPE type) {
	return getPrefixedNameByType(type, L"CommandList");
}

// TODO:
int getPixelSizeFromFormat(DXGI_FORMAT format) {
	switch (format) {
	case DXGI_FORMAT_R8G8B8A8_UNORM:
		return 4;
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		return 4 * 4;
	default:
		return 4;
	}
}

DXGI_FORMAT getDepthFormatAsNormal(DXGI_FORMAT format) {
	switch (format) {
	case DXGI_FORMAT_D16_UNORM:
		return DXGI_FORMAT_R16_FLOAT;
	case DXGI_FORMAT_D24_UNORM_S8_UINT:
		return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
	case DXGI_FORMAT_D32_FLOAT:
		return DXGI_FORMAT_R32_FLOAT;
	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
	default:
		return format;
	}
}

WString strToWStr(const String &str) {
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	return converter.from_bytes(str);
}

} // namespace Dar