#pragma once

#include "graphics/d3d12/resource_manager.h"
#include "graphics/d3d12/texture_res.h"
#include "utils/defines.h"

using TextureId = unsigned int;
#define INVALID_TEXTURE_ID (unsigned int)(-1)

struct TextureDesc {
	String path = "";
	TextureId id = INVALID_TEXTURE_ID;
};

bool uploadTextureData(
	Vector<TextureDesc> &textureDescs,
	Dar::UploadHandle uploadHandle,
	Vector<Dar::TextureResource> &textures,
	Dar::HeapHandle &texturesHeap,
	// TODO: just don't generate mips for texture that do not need them :)
	bool forceNoMips // Ignore any mip-maps if present
);