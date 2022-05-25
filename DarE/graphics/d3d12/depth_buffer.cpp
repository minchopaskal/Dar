#include "d3d12/depth_buffer.h"
#include "d3d12/resource_manager.h"
#include "utils/utils.h"

namespace Dar {

bool DepthBuffer::init(ComPtr<ID3D12Device> device, int width, int height, DXGI_FORMAT format) {
	dassert(format == DXGI_FORMAT_D16_UNORM
		|| format == DXGI_FORMAT_D24_UNORM_S8_UINT
		|| format == DXGI_FORMAT_D32_FLOAT
		|| format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT
	);

	if (!dsvHeap) {
		dsvHeap.init(
			device.Get(),
			D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
			1, /* numDescriptors */
			false /* shaderVisible */
		);
	}

	TextureInitData resData = {};
	resData.width = width;
	resData.height = height;
	resData.format = format;
	resData.clearValue.depthStencil.depth = 1.f;
	resData.clearValue.depthStencil.stencil = 0;
	
	depthTex.init(resData, TextureResourceType::DepthStencil);

	dsvHeap.reset();
	dsvHeap.addDSV(getBufferResource(), format);

	return true;
}

DXGI_FORMAT DepthBuffer::getFormatAsDepthBuffer() const {
	return depthTex.getFormat();
}

DXGI_FORMAT DepthBuffer::getFormatAsTexture() const {
	return getDepthFormatAsNormal(depthTex.getFormat());
}

D3D12_CPU_DESCRIPTOR_HANDLE DepthBuffer::getCPUHandle() const {
	return dsvHeap.getCPUHandle(0);
}

ID3D12Resource *DepthBuffer::getBufferResource() {
	return depthTex.getBufferResource();
}

ResourceHandle DepthBuffer::getBufferHandle() const {
	return depthTex.getHandle();
}

} // namespace Dar