#pragma once

#include "d3d12_defines.h"

enum class AssetType : int {
	shader = 0,
};

WString getAssetFullPath(const wchar_t *assetName, AssetType assetType);
