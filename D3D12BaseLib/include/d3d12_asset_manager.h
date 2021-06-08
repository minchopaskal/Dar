#pragma once

#include <string>

enum class AssetType : int {
	shader = 0,
};

std::wstring getAssetFullPath(const wchar_t *assetName, AssetType assetType);
