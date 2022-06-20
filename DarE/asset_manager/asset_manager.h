#pragma once

#include "utils/defines.h"

namespace Dar {

enum class AssetType {
	Shader,
	Texture,
	Scene,

	Count
};

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType);

} // namespace Dar