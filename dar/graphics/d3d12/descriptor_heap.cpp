#include "d3d12/descriptor_heap.h"
#include "utils/defines.h"
#include "utils/utils.h"

namespace Dar {

DescriptorHeap::DescriptorHeap() :
	heap{ nullptr },
	device(nullptr),
	type(D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES),
	cpuHandleStart{ 0 },
	gpuHandleStart{ 0 },
	cpuHandleRunning{ 0 },
	handleIncrementSize(0)
{ }

void DescriptorHeap::init(ID3D12Device *d, D3D12_DESCRIPTOR_HEAP_TYPE t, UINT numDesctiptors, bool shaderVisible) {
	dassert(d != nullptr);
	deinit();

	device = d;
	type = t;

	const bool isRTVOrDSV = (type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV || type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
	shaderVisible = (shaderVisible && !isRTVOrDSV);

	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.Type = type;
	desc.NumDescriptors = numDesctiptors;
	desc.Flags = (shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
	desc.NodeMask = 0;

	RETURN_ON_ERROR(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)), , "Failed to create descriptor heap!");

	cpuHandleRunning = cpuHandleStart = heap->GetCPUDescriptorHandleForHeapStart();
	gpuHandleStart = shaderVisible ? heap->GetGPUDescriptorHandleForHeapStart() : D3D12_GPU_DESCRIPTOR_HANDLE{};

	handleIncrementSize = device->GetDescriptorHandleIncrementSize(type);
}

void DescriptorHeap::reset() {
	cpuHandleRunning = cpuHandleStart;
}

void DescriptorHeap::addTexture2DSRV(const TextureResource &tex) {
	dassert(type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
	desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	desc.Format = getDepthFormatAsNormal(tex.getFormat());
	desc.Texture2D.MipLevels = tex.getNumMipLevels();
	desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	device->CreateShaderResourceView(tex.getBufferResource(), &desc, cpuHandleRunning);
	cpuHandleRunning.ptr += handleIncrementSize;
}

void DescriptorHeap::addTextureCubeSRV(const TextureResource &tex) {
	dassert(type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	dassert(tex.getDepth() == 6);

	D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
	desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
	desc.Format = getDepthFormatAsNormal(tex.getFormat());
	desc.TextureCube.MipLevels = tex.getNumMipLevels();
	desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	device->CreateShaderResourceView(tex.getBufferResource(), &desc, cpuHandleRunning);
	cpuHandleRunning.ptr += handleIncrementSize;
}

void DescriptorHeap::addBufferSRV(ID3D12Resource *resource, UINT numElements, UINT elementSize) {
	dassert(type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
	desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	desc.Buffer.FirstElement = 0;
	desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	desc.Buffer.NumElements = numElements;
	desc.Buffer.StructureByteStride = elementSize;

	device->CreateShaderResourceView(resource, &desc, cpuHandleRunning);
	cpuHandleRunning.ptr += handleIncrementSize;
}

void DescriptorHeap::addBufferUAV(ID3D12Resource *resource, UINT numElements, UINT elementSize) {
	dassert(type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
	desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	desc.Buffer.FirstElement = 0;
	desc.Buffer.NumElements = numElements;
	desc.Buffer.StructureByteStride = elementSize;
	desc.Buffer.CounterOffsetInBytes = 0;
	desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

	device->CreateUnorderedAccessView(resource, nullptr, &desc, cpuHandleRunning);
	
	cpuHandleRunning.ptr += handleIncrementSize;
}

void DescriptorHeap::addRTV(ID3D12Resource *resource, D3D12_RENDER_TARGET_VIEW_DESC *rtvDesc) {
	dassert(type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	device->CreateRenderTargetView(resource, rtvDesc, cpuHandleRunning);
	cpuHandleRunning.ptr += handleIncrementSize;
}

void DescriptorHeap::addDSV(ID3D12Resource *resource, DXGI_FORMAT format) {
	dassert(type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

	D3D12_DEPTH_STENCIL_VIEW_DESC dsDesc = {};
	dsDesc.Format = format;
	dsDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsDesc.Flags = D3D12_DSV_FLAG_NONE;
	dsDesc.Texture2D.MipSlice = 0;

	device->CreateDepthStencilView(resource, &dsDesc, cpuHandleRunning);
	cpuHandleRunning.ptr += handleIncrementSize;
}

void DescriptorHeap::deinit() {
	heap.Reset();
	device = nullptr;
	cpuHandleStart = cpuHandleRunning = { 0 };
	gpuHandleStart = { 0 };
	handleIncrementSize = 0;

}

} // namespace Dar