#include "asset_manager.h"

std::wstring getAssetFullPath(const wchar_t *assetName, AssetType assetType) {
	std::wstring path;

	switch (assetType) {
	case AssetType::shader:
		path = LR"(.\res\shaders\)";
		break;
	default:
		return assetName;
	}

	path.append(assetName);
	return path;
}
