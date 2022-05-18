#pragma once

#include "d3d12/resource_handle.h"
#include "d3d12/includes.h"
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
	~TextureResource();

	/// Initialize the texture resource. Destroys the underlying texture resource if any.
	/// @param initData Texture initialization data.
	/// @param heapInfo Description of a heap and its place in it if we want to use it.
	///                 If null the texture is created normally as a commited resource.
	/// @param type Type of the texture.
	/// @return true on success, false otherwise.
	bool init(TextureInitData &initData, TextureResourceType type, HeapInfo *heapInfo = nullptr);

	void setName(const WString &name);

	WString getName() const;

	ID3D12Resource *getBufferResource() const {
		return handle.get();
	}

	ResourceHandle getHandle() const {
		return handle;
	}

	DXGI_FORMAT getFormat() const {
		return format;
	}

	int getWidth() const {
		return width;
	}

	int getHeight() const {
		return height;
	}

	void deinit();

private:
	ResourceHandle handle = INVALID_RESOURCE_HANDLE; ///< Handle to the resource that one can pass to the ResourceManager methods.
	WString name;
	DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
	int width = 0;
	int height = 0;
};

} // namespace Dar
