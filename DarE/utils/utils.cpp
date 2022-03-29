#include "utils/utils.h"

WString getPrefixedNameByType(D3D12_COMMAND_LIST_TYPE type, LPWSTR prefix) {
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
