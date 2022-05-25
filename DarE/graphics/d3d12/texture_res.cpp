#include "d3d12/texture_res.h"

#include "d3d12/resource_manager.h"
#include "utils/utils.h"

namespace Dar {

bool TextureResource::init(TextureInitData &texInitData, TextureResourceType type, HeapInfo *heapInfo) {
	ResourceInitData resInitData;
	switch (type) {
	case TextureResourceType::ShaderResource:
		resInitData.init(ResourceType::TextureBuffer);
		break;
	case TextureResourceType::RenderTarget:
		resInitData.init(ResourceType::RenderTargetBuffer);
		break;
	case TextureResourceType::DepthStencil:
		resInitData.init(ResourceType::DepthStencilBuffer);
		break;
	default:
		return false;
	}

	resInitData.heapInfo = heapInfo;
	resInitData.textureData = texInitData;

	deinit();

	auto &resManager = getResourceManager();
	handle = resManager.createBuffer(resInitData);

	const bool success = handle != INVALID_RESOURCE_HANDLE;

	if (success) {
		texData = texInitData;
	}

	return success;
}

UINT64 TextureResource::upload(UploadHandle uploadHandle, const void *data) {
	if (handle == INVALID_RESOURCE_HANDLE) {
		return 0;
	}

	auto &resManager = getResourceManager();

	D3D12_SUBRESOURCE_DATA textureSubresources = {};
	textureSubresources.pData = data;
	textureSubresources.RowPitch = static_cast<UINT64>(texData.width) * static_cast<UINT64>(getPixelSizeFromFormat(texData.format));
	textureSubresources.SlicePitch = textureSubresources.RowPitch * texData.height;
	return resManager.uploadTextureData(uploadHandle, handle, &textureSubresources, 1, 0);
}

void TextureResource::setName(const WString &name) {
	if (handle == INVALID_RESOURCE_HANDLE) {
		return;
	}

	auto &resManager = getResourceManager();

	resManager.getID3D12Resource(handle)->SetName(name.c_str());

	this->name = name;
}

WString TextureResource::getName() const {
	return name;
}

void TextureResource::deinit() {
	if (handle == INVALID_RESOURCE_HANDLE) {
		return;
	}

	auto &resManager = getResourceManager();

	if (handle != INVALID_RESOURCE_HANDLE) {
		resManager.deregisterResource(handle);
	}

	texData = {};
}

} // namespace Dar