#include "d3d12/resource_manager.h"

#include "AgilitySDK/include/d3d12.h"

#include "d3dx12.h"

#define CHECK_RESOURCE_HANDLE(handle) \
do { \
	if (handle == INVALID_RESOURCE_HANDLE || handle >= resources.size() || resources[handle].res == nullptr) { \
		return false; \
	} \
} while (false)

#define CHECK_HEAP_HANDLE(handle) \
do { \
	if (handle == INVALID_HEAP_HANDLE || handle >= heaps.size() || heaps[handle].heap == nullptr) { \
		return false; \
	} \
} while (false)

WString getResourceNameByType(ResourceType type) {
	switch (type) {
	case ResourceType::DataBuffer:
		return L"DataBuffer";
	case ResourceType::StagingBuffer:
		return L"StagingBuffer";
	case ResourceType::DepthStencilBuffer:
		return L"DepthStencilBuffer";
	case ResourceType::TextureBuffer:
		return L"TextureBuffer";
	default:
		return L"UnknownTypeBuffer";
	}
}

HeapHandle ResourceManager::createHeap(SizeType size, HeapAlignmentType alignment) {
	D3D12_HEAP_PROPERTIES heapProps = {};
	heapProps.Type = D3D12_HEAP_TYPE_DEFAULT; // TODO: think about upload/readback heap creation methods.

	D3D12_HEAP_DESC heapDesc = {};
	heapDesc.SizeInBytes = size;
	heapDesc.Alignment = static_cast<UINT64>(alignment);
	heapDesc.Flags = D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES;
	heapDesc.Properties = heapProps;

	HeapHandle handle = INVALID_HEAP_HANDLE;
	if (!heapHandlesPool.empty()) {
		handle = heapHandlesPool.front();
		heapHandlesPool.pop();
	} else {
		heaps.emplace_back();
		handle = static_cast<SizeType>(heaps.size() - 1);
		heaps[handle].alignment = alignment;
		heaps[handle].size = size;
	}

	HRESULT err = device->CreateHeap(&heapDesc, IID_PPV_ARGS(heaps[handle].heap.GetAddressOf()));
	if (err != S_OK) {
		deallocateHeap(handle);
		return INVALID_HEAP_HANDLE;
	}

	return handle;
}

HeapHandle ResourceManager::createHeap(D3D12_RESOURCE_DESC *resDescriptors, UINT numResources) {
	D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, numResources, resDescriptors);
	return createHeap(static_cast<SizeType>(allocInfo.SizeInBytes), static_cast<HeapAlignmentType>(allocInfo.Alignment));
}

void ResourceManager::createHeap(D3D12_RESOURCE_DESC *resDescriptors, UINT numResources, HeapHandle &handle) {
	D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, numResources, resDescriptors);
	HeapAlignmentType alignment = static_cast<HeapAlignmentType>(allocInfo.Alignment);
	SizeType size = static_cast<SizeType>(allocInfo.SizeInBytes);
	bool canReuse =
		isValidHeapHandle(handle) &&
		(alignment == getHeapAlignment(handle)) &&
		(size <= getHeapSize(handle));

	if (!canReuse) {
		deallocateHeap(handle);
		handle = createHeap(size, alignment);
	}
}

bool ResourceManager::deallocateHeap(HeapHandle &handle) {
	CHECK_HEAP_HANDLE(handle);

	heaps[handle].heap.Reset();
	heaps[handle].size = 0;

	heapHandlesPool.push(handle);

	handle = INVALID_HEAP_HANDLE;
}

ResourceHandle ResourceManager::createBuffer(ResourceInitData &initData) {
	ResourceType type = initData.type;

	if (type == ResourceType::StagingBuffer) {
		CHECK_RESOURCE_HANDLE(initData.stagingData.destResource);
	}

	D3D12_RESOURCE_STATES initialState = initData.state;
	D3D12_CLEAR_VALUE clearValue = {};
	if (type != ResourceType::DataBuffer && type != ResourceType::StagingBuffer) {
		clearValue.Format = initData.textureData.format;
		if (type == ResourceType::DepthStencilBuffer) {
			clearValue.DepthStencil.Depth = initData.textureData.clearValue.depthStencil.depth;
			clearValue.DepthStencil.Stencil = initData.textureData.clearValue.depthStencil.stencil;
		} else {
			for (int i = 0; i < 4; ++i) {
				clearValue.Color[i] = initData.textureData.clearValue.color[i];
			}
		}
	}
	SizeType resourceSize = 0;

	ComPtr<ID3D12Resource> resource;
	D3D12_HEAP_PROPERTIES heapProps = {};
	D3D12_CLEAR_VALUE *clearValuePtr = nullptr;
	D3D12_RESOURCE_DESC resDesc = initData.getResourceDescriptor();
	switch (type) {
	case ResourceType::DataBuffer:
		heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		break;
	case ResourceType::TextureBuffer:
		heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		break;
	case ResourceType::RenderTargetBuffer:
		initialState = D3D12_RESOURCE_STATE_COMMON;
		heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		clearValuePtr = &clearValue;
		break;
	case ResourceType::DepthStencilBuffer:
		initialState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
		heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		clearValuePtr = &clearValue;
		break;
	case ResourceType::StagingBuffer:
		heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
		break;
	default:
		return INVALID_RESOURCE_HANDLE;
	}

	bool placedResource = false;
	if (initData.heapInfo != nullptr && isValidHeapHandle(initData.heapInfo->handle)) { // We have a heap on which to create the placed resource
		HeapHandle heapHandle = initData.heapInfo->handle;
		SizeType offset = initData.heapInfo->offset;
		SizeType alignment = static_cast<SizeType>(heaps[heapHandle].alignment);
		dassert(offset % alignment == 0);
		if (offset % alignment != 0) { // make sure to pass valid offset, so we don't crash
			offset = (offset + alignment - 1) / alignment;
		}

		if (initData.size + offset < getHeapSize(heapHandle)) {
			HRESULT res = device->CreatePlacedResource(
				getID3D12Heap(heapHandle),
				offset,
				&resDesc,
				initialState,
				clearValuePtr,
				IID_PPV_ARGS(resource.GetAddressOf())
			);

			if (res == S_OK) {
				placedResource = true;
			} else {
				auto err = GetLastError();
				LOG_FMT(Warning, "Failed to create placed resource with error: %lu! Falling back on creating a commited resource!\n", err);
			}
		}
	}
	
	if (!placedResource) {
		RETURN_ON_ERROR(
			device->CreateCommittedResource(
				&heapProps,
				initData.heapFlags,
				&resDesc,
				initialState,
				clearValuePtr,
				IID_PPV_ARGS(resource.GetAddressOf())
			),
			INVALID_RESOURCE_HANDLE,
			"Failed to create committed resource!"
		);
	}

	WString name = initData.name.size() == 0 ? getResourceNameByType(type) : initData.name;
	resource->SetName(name.c_str());

	D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, 1, &resDesc);
	SizeType size = allocInfo.SizeInBytes;

	const UINT numSubresources = type == ResourceType::TextureBuffer && initData.textureData.mipLevels > 0 ? initData.textureData.mipLevels : 1;
#ifdef D3D12_NDEBUG
	return registerResource(
		resource,
		numSubresources,
		initialState
	);
#else
	return registerResource(
		resource,
		numSubresources,
		size,
		initialState,
		type
	);
#endif // D3D12_NDEBUG
}

UploadHandle ResourceManager::beginNewUpload() {
	cmdLists.emplace_back(copyQueue.getCommandList());
	return cmdLists.size() - 1;
}

bool ResourceManager::uploadBufferData(UploadHandle uploadHandle, ResourceHandle destResourceHandle, const void *data, SizeType size) {
	CHECK_RESOURCE_HANDLE(destResourceHandle);

	ResourceInitData resData(ResourceType::StagingBuffer);
	resData.stagingData.destResource = destResourceHandle;

	ResourceHandle stagingBufferHandle = createBuffer(resData);
	stagingBuffers.push_back(stagingBufferHandle);

	ID3D12Resource *destResource = getID3D12Resource(destResourceHandle);
	ID3D12Resource *stageResource = getID3D12Resource(stagingBufferHandle);

#ifdef D3D12_DEBUG
	SubresStates states;
	getLastGlobalState(destResourceHandle, states);
	dassert(states.size() == 1);
#endif // D3D12_DEBUG

	UINT8 *mappedBuffer = nullptr;
	D3D12_RANGE range = { 0, 0 };
	stageResource->Map(0, &range, reinterpret_cast<void**>(&mappedBuffer));
	memcpy(mappedBuffer, data, size);
	stageResource->Unmap(0, &range);

	cmdLists[uploadHandle]->CopyBufferRegion(destResource, 0, stageResource, 0, size);

	return true;
}

UINT64 ResourceManager::uploadTextureData(UploadHandle uploadHandle, ResourceHandle destResourceHandle, D3D12_SUBRESOURCE_DATA *subresData, UINT numSubresources, UINT startSubresourceIndex) {
	CHECK_RESOURCE_HANDLE(destResourceHandle);

	ResourceInitData resData(ResourceType::StagingBuffer);
	resData.stagingData.destResource = destResourceHandle;
	resData.stagingData.firstSubresourceIndex = startSubresourceIndex;
	resData.stagingData.numSubresources = numSubresources;

	ResourceHandle stagingBufferHandle = createBuffer(resData);
	stagingBuffers.push_back(stagingBufferHandle);
	ID3D12Resource *destResource = getID3D12Resource(destResourceHandle);
	ID3D12Resource *stageResource = getID3D12Resource(stagingBufferHandle);

#ifdef D3D12_DEBUG
	SubresStates states;
	getLastGlobalState(destResourceHandle, states);
	dassert(states.size() >= numSubresources + startSubresourceIndex);
#endif // D3D12_DEBUG

	return UpdateSubresources(cmdLists[uploadHandle].get(), destResource, stageResource, 0, startSubresourceIndex, numSubresources, subresData);
}

bool ResourceManager::uploadBuffers() {
	for (int i = INVALID_UPLOAD_HANDLE + 1; i < cmdLists.size(); ++i) {
		copyQueue.addCommandListForExecution(std::move(cmdLists[i]));
	}
	resetCommandLists();

	UINT64 fence = copyQueue.executeCommandLists();
	copyQueue.waitForFenceValue(fence);

	return true;
}

bool ResourceManager::flush() {
	return uploadBuffers();
}

unsigned int ResourceManager::getSubresourcesCount(ResourceHandle handle) {
	CHECK_RESOURCE_HANDLE(handle);

	return (unsigned int)(resources[handle].subresStates.size());
}

bool ResourceManager::getLastGlobalState(ResourceHandle handle, SubresStates &outStates) {
	CHECK_RESOURCE_HANDLE(handle);

	CriticalSectionLock lock = resources[handle].cs.lock();
	SubresStates &states = resources[handle].subresStates;
	outStates.resize(states.size());
	for (int i = 0; i < states.size(); ++i) {
		outStates[i] = states[i];
	}

	return true;
}

bool ResourceManager::getLastGlobalStateForSubres(ResourceHandle handle, D3D12_RESOURCE_STATES &outState, const unsigned int subresIndex) {
	CHECK_RESOURCE_HANDLE(handle);

	// TODO: Experiment with locking individual subresources
	CriticalSectionLock lock = resources[handle].cs.lock();
	SubresStates &states = resources[handle].subresStates;
	if (subresIndex < 0 || states.size() <= subresIndex) {
		return false;
	}

	outState = states[subresIndex];
	return true;
}

bool ResourceManager::setGlobalState(ResourceHandle handle, const D3D12_RESOURCE_STATES &state) {
	CHECK_RESOURCE_HANDLE(handle);
	
	CriticalSectionLock lock = resources[handle].cs.lock();
	SubresStates &states = resources[handle].subresStates;
	for (int i = 0; i < states.size(); ++i) {
		states[i] = state;
	}

	return true;
}

bool ResourceManager::setGlobalStateForSubres(ResourceHandle handle, const D3D12_RESOURCE_STATES &state, const unsigned int subresIndex) {
	CHECK_RESOURCE_HANDLE(handle);

	CriticalSectionLock lock = resources[handle].cs.lock();
	SubresStates &states = resources[handle].subresStates;
	if (subresIndex < 0 || states.size() <= subresIndex) {
		return false;
	}
	states[subresIndex] = state;

	return true;
}

ResourceHandle ResourceManager::registerResourceImpl(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state) {
	ResourceHandle handle;
	auto lock = resourcesCS.lock();
	if (!resourcePool.empty()) {
		handle = resourcePool.front();
		resourcePool.pop();
		// No need to make a new 
		resources[handle].res = resourcePtr;
		resources[handle].subresStates = SubresStates{ subresourcesCount, state };
		resources[handle].size = size;
	} else {
		resources.push_back(Resource{ resourcePtr, SubresStates(subresourcesCount, state), {}, size });
		handle = resources.size() - 1;
		if (numThreads > 1) {
			resources[handle].cs.init();
		}
	}

	return handle;
}

#ifdef D3D12_DEBUG
ResourceHandle ResourceManager::registerResource(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state, ResourceType type) {
	ResourceHandle handle = registerResourceImpl(resourcePtr, subresourcesCount, size, state);
	resources[handle].type = type;
	return handle;
}
#else
ResourceHandle ResourceManager::registerResource(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state) {
	return registerResourceImpl(resourcePtr, subresourcesCount, size, state);
}
#endif // D3D12_DEBUG

bool ResourceManager::deregisterResource(ResourceHandle &handle) {
	CHECK_RESOURCE_HANDLE(handle);

	unsigned long refCount = resources[handle].res.Reset();

#ifdef D3D12_DEBUG
	ResourceType type = getResourceType(handle);
	dassert(type != ResourceType::Invalid);
	if (type == ResourceType::StagingBuffer) {
		dassert(refCount == 0);
	}
#endif // D3D12_DEBUG

	{
		auto lock = resourcesCS.lock();
		resourcePool.push(handle);
	}

	handle = INVALID_RESOURCE_HANDLE;
	return true;
}

ID3D12Resource* ResourceManager::getID3D12Resource(ResourceHandle handle) {
	CHECK_RESOURCE_HANDLE(handle);

	return resources[handle].res.Get();
}

ID3D12Heap *ResourceManager::getID3D12Heap(HeapHandle handle) {
	CHECK_HEAP_HANDLE(handle);
	return heaps[handle].heap.Get();
}

HeapAlignmentType ResourceManager::getHeapAlignment(HeapHandle handle) {
	if (!isValidHeapHandle(handle)) {
		return HeapAlignmentType::Invalid;
	}
	return heaps[handle].alignment;
}

SizeType ResourceManager::getHeapSize(HeapHandle handle) {
	CHECK_HEAP_HANDLE(handle);
	return heaps[handle].size;
}

void ResourceManager::endFrame() {
	// Destroy any staging resources
	for (int i = 0; i < stagingBuffers.size(); ++i) {
		deregisterResource(stagingBuffers[i]);
	}
}

SizeType ResourceManager::getResourceSize(ResourceHandle handle) const {
	if (handle == INVALID_RESOURCE_HANDLE) {
		return 0;
	}

	return resources[handle].size;
}

void ResourceManager::resetCommandLists() {
	cmdLists.clear();
}

bool ResourceManager::isValidHeapHandle(HeapHandle handle) const {
	return !(handle == INVALID_HEAP_HANDLE || handle >= heaps.size() || heaps[handle].heap == nullptr);
}

#ifdef D3D12_DEBUG
ResourceType ResourceManager::getResourceType(ResourceHandle handle) {
	return resources[handle].type;
}
#endif // D3D12_DEBUG

ResourceManager *g_ResourceManager = nullptr;

bool initResourceManager(ComPtr<ID3D12Device8> device, const unsigned int nt) {
	deinitResourceManager();

	g_ResourceManager = new ResourceManager;
	g_ResourceManager->device = device;
	g_ResourceManager->copyQueue.init(device);
	if (nt > 1) {
		g_ResourceManager->numThreads = nt;
		return g_ResourceManager->resourcesCS.init();
	}

	// Create a sentinel resource having index of INVALID_RESOURCE_HANDLE
	g_ResourceManager->resources.emplace_back();

	// Same for INVALID_UPLOAD_HANDLE
	g_ResourceManager->resetCommandLists();

	return true;
}

void deinitResourceManager() {
	if (g_ResourceManager) {
		delete g_ResourceManager;
		g_ResourceManager = nullptr;
	}
}

ResourceManager& getResourceManager() {
	if (g_ResourceManager == nullptr) {
		dassert(false);
	}

	return *g_ResourceManager;
}

D3D12_RESOURCE_DESC ResourceInitData::getResourceDescriptor() {
	D3D12_RESOURCE_DESC resDesc = {};
	SizeType resourceSize = 0;
	switch (type) {
	case ResourceType::DataBuffer:
		resDesc = CD3DX12_RESOURCE_DESC::Buffer(size);
		break;
	case ResourceType::TextureBuffer:
		resDesc = CD3DX12_RESOURCE_DESC::Tex2D(
			textureData.format,
			textureData.width,
			textureData.height,
			1, /* arraySize */
			textureData.mipLevels,
			textureData.samplesCount,
			textureData.samplesQuality
		);
		break;
	case ResourceType::RenderTargetBuffer:
		resDesc = CD3DX12_RESOURCE_DESC::Tex2D(
			textureData.format,
			textureData.width,
			textureData.height,
			1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
		);
		break;
	case ResourceType::DepthStencilBuffer:
		resDesc = CD3DX12_RESOURCE_DESC::Tex2D(
			textureData.format,
			textureData.width,
			textureData.height,
			1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
		);
		break;
	case ResourceType::StagingBuffer:
		resourceSize = GetRequiredIntermediateSize(
			stagingData.destResource.get(),
			stagingData.firstSubresourceIndex,
			stagingData.numSubresources
		);
		resDesc = CD3DX12_RESOURCE_DESC::Buffer(resourceSize);
		break;
	default:
		break;
	}

	return resDesc;
}
