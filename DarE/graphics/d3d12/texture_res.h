#pragma once

#include "d3d12/includes.h"
#include "d3d12/resource_manager.h"
#include "utils/defines.h"

namespace Dar {

struct TextureInitData;
struct HeapInfo;

enum class TextureResourceType {
	ShaderResource,
	RenderTarget,
	DepthStencil
};

/// A wrapper around ResourceHandle.
/// This is the preferred way of creating a texture
/// since it carries a semantic meaning. Otherwise
/// one can create a texture through ResourceManager::createBuffer
/// and save the resource handle to it.
struct TextureResource {
	TextureResource() {}
	
	/// Initialize the texture resource. Destroys the underlying texture resource if any.
	/// @param initData Texture initialization data.
	/// @param heapInfo Description of a heap and its place in it if we want to use it.
	///                 If null the texture is created normally as a commited resource.
	/// @param type Type of the texture.
	/// @return true on success, false otherwise.
	bool init(TextureInitData &initData, TextureResourceType type, HeapInfo *heapInfo = nullptr);

	/// Upload texture data to the texture resource
	/// @param uploadHandle Handle with which the upload manager uploads the resource.
	/// @param data The texture data to be uploaded to the GPU.
	/// @return Size in bytes of the uploaded texture. 0 on fail.
	UINT64 upload(UploadHandle uploadHandle, const void *data);

	void setName(const WString &name);

	WString getName() const;

	ID3D12Resource *getBufferResource() const {
		return handle.get();
	}

	ResourceHandle getHandle() const {
		return handle;
	}

	DXGI_FORMAT getFormat() const {
		return texData.format;
	}

	int getWidth() const {
		return texData.width;
	}

	int getHeight() const {
		return texData.height;
	}

	int getNumMipLevels() const {
		return texData.mipLevels;
	}

	void deinit();

private:
	ResourceHandle handle = INVALID_RESOURCE_HANDLE; ///< Handle to the resource that one can pass to the ResourceManager methods.
	WString name;
	TextureInitData texData;
};

} // namespace Dar
