#pragma once

#include "d3d12/descriptor_heap.h"
#include "d3d12/includes.h"
#include "d3d12/resource_handle.h"
#include "d3d12/texture_res.h"
#include "utils/utils.h"

namespace Dar {

struct DepthBuffer {
	/// Intialize or resize the depth buffer.
	/// @param device Device used for creation of the DSV heap.
	/// @param width Width of the depth buffer texture
	/// @param height Height of the depth buffer texture
	/// @param format Format of the depth buffer. Must be one of DXGI_FORMAT_D* types.
	/// @return true on success, false otherwise
	bool init(ComPtr<ID3D12Device> device, int width, int height, DXGI_FORMAT format, const String& name);

	/// @return Format of the depth buffer when used as a depth attachment.
	DXGI_FORMAT getFormatAsDepthBuffer() const;

	/// @return Format of the depth buffer when used  as a texture.
	DXGI_FORMAT getFormatAsTexture() const;

	D3D12_CPU_DESCRIPTOR_HANDLE getCPUHandle() const;

	ID3D12Resource *getBufferResource();

	ResourceHandle getBufferHandle() const;

	const TextureResource& getTexture() const {
		return depthTex;
	}

	void setName(const String &name) {
		getBufferResource()->SetName(strToWStr(name).c_str());
	}

private:
	DescriptorHeap dsvHeap;
	TextureResource depthTex;
};

} // namespace Dar