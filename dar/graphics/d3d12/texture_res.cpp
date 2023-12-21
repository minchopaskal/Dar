#include "d3d12/texture_res.h"

#include "d3d12/resource_manager.h"
#include "utils/utils.h"

#include "reslib/img_data.h"

namespace Dar {

bool TextureResource::init(TextureInitData &texInitData, TextureResourceType type, const String &resName, Optional<HeapHandle> heapHandle) {
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

	resInitData.heapHandle = heapHandle;
	resInitData.textureData = texInitData;
	resInitData.name = resName;

	deinit();

	auto &resManager = getResourceManager();
	handle = resManager.createBuffer(resInitData);

	const bool success = handle != INVALID_RESOURCE_HANDLE;

	if (success) {
		texData = texInitData;
		name = resName;
	}

	return success;
}

UINT64 TextureResource::upload(UploadHandle uploadHandle, ImageData &imgData) {
	static constexpr SizeType BC7_BYTES_PER_BLOCK = 16;

	if (handle == INVALID_RESOURCE_HANDLE) {
		return 0;
	}

	Vector<D3D12_SUBRESOURCE_DATA> textureSubresources;
	int width = imgData.header.width;
	int height = imgData.header.height;
	for (int i = 0; i < imgData.header.mipMapCount; ++i) {
		SizeType offset = imgData.header.mipOffsets[i];

		D3D12_SUBRESOURCE_DATA subresData = {};
		subresData.pData = reinterpret_cast<void*>(imgData.data + offset);

		// BC3/BC7 block is 4x4 so each row would have width/4 blocks
		SizeType numBlocksPerRow = std::max(1, (width + 3) / 4);
		SizeType numBlocksPerColumn = std::max(1, (height + 3) / 4);

		subresData.RowPitch = numBlocksPerRow * BC7_BYTES_PER_BLOCK;
		subresData.SlicePitch = subresData.RowPitch * numBlocksPerColumn;

		textureSubresources.push_back(subresData);

		width /= 2;
		height /= 2;
	}
	
	auto& resManager = getResourceManager();
	return resManager.uploadTextureData(uploadHandle, handle, textureSubresources.data(), static_cast<UINT>(textureSubresources.size()), 0);
}

void TextureResource::setName(const String &n) {
	if (handle == INVALID_RESOURCE_HANDLE) {
		return;
	}

	auto &resManager = getResourceManager();

	resManager.getID3D12Resource(handle)->SetName(strToWStr(n).c_str());

	name = n;
}

String TextureResource::getName() const {
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