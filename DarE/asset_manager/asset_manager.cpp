#include "asset_manager/asset_manager.h"

#ifdef DAR_DEBUG
#include <filesystem>
#include <cassert>
#endif DAR_DEBUG

namespace Dar {

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType) {
	WString path;

	switch (assetType) {
	case AssetType::Shader:
		path = LR"(.\res\shaders\)";
		break;
	case AssetType::Texture:
		path = LR"(.\res\textures\)";
		break;
	case AssetType::Scene:
		path = LR"(.\res\scenes\)";
		break;
	default:
		return assetName;
	}

	path.append(assetName);

#ifdef DAR_DEBUG
	std::filesystem::path fsPath(path);
	assert(std::filesystem::exists(fsPath));
#endif DAR_DEBUG

	return path;
}

} // namespace Dar
