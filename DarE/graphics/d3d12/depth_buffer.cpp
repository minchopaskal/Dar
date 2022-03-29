#include "d3d12/depth_buffer.h"
#include "d3d12/resource_manager.h"

bool DepthBuffer::init(ComPtr<ID3D12Device> device, int width, int height, DXGI_FORMAT format) {
	dassert(format == DXGI_FORMAT_D16_UNORM
		|| format == DXGI_FORMAT_D24_UNORM_S8_UINT
		|| format == DXGI_FORMAT_D32_FLOAT
		|| format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT
	);

	this->format = format;

	if (!dsvHeap) {
		dsvHeap.init(
			device.Get(),
			D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
			1, /* numDescriptors */
			false /* shaderVisible */
		);
	}

	ResourceManager &resManager = getResourceManager();

	resManager.deregisterResource(bufferHandle);

	ResourceInitData resData(ResourceType::DepthStencilBuffer);
	resData.textureData.width = width;
	resData.textureData.height = height;
	resData.textureData.format = format;
	bufferHandle = resManager.createBuffer(resData);

	dsvHeap.reset();
	dsvHeap.addDSV(bufferHandle.get(), format);

	return true;
}

DXGI_FORMAT DepthBuffer::getFormatAsDepthBuffer() const {
	return format;
}

DXGI_FORMAT DepthBuffer::getFormatAsTexture() const {
	switch (format) {
	case DXGI_FORMAT_D16_UNORM:
		return DXGI_FORMAT_R16_FLOAT;
	case DXGI_FORMAT_D24_UNORM_S8_UINT:
		return DXGI_FORMAT_R24G8_TYPELESS;
	case DXGI_FORMAT_D32_FLOAT:
		return DXGI_FORMAT_R32_FLOAT;
	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
	default:
		return DXGI_FORMAT_UNKNOWN;
	}
}

D3D12_CPU_DESCRIPTOR_HANDLE DepthBuffer::getCPUHandle() const {
	return dsvHeap.getCPUHandle(0);
}

ID3D12Resource* DepthBuffer::getBufferResource() {
	return bufferHandle.get();
}

ResourceHandle DepthBuffer::getBufferHandle() const {
	return bufferHandle;
}
