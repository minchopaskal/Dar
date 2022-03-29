#include "asset_manager/asset_manager.h"

#ifdef D3D12_DEBUG
#include <filesystem>
#include <cassert>
#endif D3D12_DEBUG

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType) {
	WString path;

	switch (assetType) {
	case AssetType::Shader:
		path = LR"(.\res\shaders\)";
		break;
	case AssetType::Texture:
		path = LR"(.\res\textures\)";
		break;
	default:
		return assetName;
	}

	path.append(assetName);

#ifdef D3D12_DEBUG
	std::filesystem::path fsPath(path);
	assert(std::filesystem::exists(fsPath));
#endif D3D12_DEBUG

	return path;
}
