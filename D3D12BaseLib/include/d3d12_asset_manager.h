#pragma once

#include "d3d12_defines.h"

enum class AssetType {
	Shader,
	Texture,
};

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType);
