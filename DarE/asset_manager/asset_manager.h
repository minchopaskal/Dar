#pragma once

#include "utils/defines.h"

enum class AssetType {
	Shader,
	Texture,
};

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType);
