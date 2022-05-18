#include "d3d12/texture_res.h"

#include "d3d12/resource_manager.h"

namespace Dar {

TextureResource::~TextureResource() {
	deinit();
}

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
		format = texInitData.format;
	}

	return success;
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

	format = DXGI_FORMAT_UNKNOWN;
	name = L"";
	width = height = 0;
}

} // namespace Dar