#include "d3d12_asset_manager.h"

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType) {
	WString path;

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
