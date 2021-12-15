#include "d3d12_resource_manager.h"

#include "AgilitySDK/include/d3d12.h"

#include "d3dx12.h"

#define CHECK_RESOURCE_HANDLE(handle) \
do { \
	if (handle == INVALID_RESOURCE_HANDLE || handle >= resources.size() || resources[handle].res == nullptr) { \
		return INVALID_RESOURCE_HANDLE; \
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

ResourceHandle ResourceManager::createBuffer(const ResourceInitData &initData) {
	ResourceType type = initData.type;

	if (type == ResourceType::StagingBuffer) {
		CHECK_RESOURCE_HANDLE(initData.stagingData.destResource);
	}

	D3D12_RESOURCE_STATES initialState = initData.state;
	D3D12_CLEAR_VALUE clearValue = {};
	SizeType resourceSize = 0;

	ComPtr<ID3D12Resource> resource;
	switch (type) {
	case ResourceType::DataBuffer:
		RETURN_ON_ERROR(
			device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				initData.heapFlags,
				&CD3DX12_RESOURCE_DESC::Buffer(initData.size),
				initialState,
				nullptr,
				IID_PPV_ARGS(resource.GetAddressOf())
			),
			INVALID_RESOURCE_HANDLE,
			"Failed to create DataBuffer!"
		);
		break;
	case ResourceType::TextureBuffer:
		RETURN_ON_ERROR(
			device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				initData.heapFlags,
				&CD3DX12_RESOURCE_DESC::Tex2D(
					initData.textureData.format,
					initData.textureData.width, 
					initData.textureData.height, 
					1, /* arraySize */
					initData.textureData.mipLevels, 
					initData.textureData.samplesCount, 
					initData.textureData.samplesQuality
				),
				initialState,
				nullptr,
				IID_PPV_ARGS(resource.GetAddressOf())
			),
			INVALID_RESOURCE_HANDLE,
			"Failed to create TextureBuffer!"
		);
		break;
	case ResourceType::DepthStencilBuffer:
		initialState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
		clearValue.Format = DXGI_FORMAT_D32_FLOAT;
		clearValue.DepthStencil = { 1.f, 0 };
		RETURN_ON_ERROR(
			device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				initData.heapFlags,
				&CD3DX12_RESOURCE_DESC::Tex2D(
					DXGI_FORMAT_D32_FLOAT,
					initData.textureData.width,
					initData.textureData.height,
					1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
				),
				initialState,
				&clearValue,
				IID_PPV_ARGS(resource.GetAddressOf())
			),
			INVALID_RESOURCE_HANDLE,
			"Failed to create/resize depth buffer!"
		);
		break;
	case ResourceType::StagingBuffer:
		resourceSize = GetRequiredIntermediateSize(
			resources[initData.stagingData.destResource].res.Get(),
			initData.stagingData.firstSubresourceIndex,
			initData.stagingData.numSubresources
		);

		RETURN_ON_ERROR(
			device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
				initData.heapFlags,
				&CD3DX12_RESOURCE_DESC::Buffer(resourceSize),
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(resource.GetAddressOf())
			),
			INVALID_RESOURCE_HANDLE,
			"Failed to create staging buffer resource!"
		);
		break;
	default:
		return INVALID_RESOURCE_HANDLE;
	}

	WString name = initData.name.size() == 0 ? getResourceNameByType(type) : initData.name;
	resource->SetName(name.c_str());

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

bool ResourceManager::uploadTextureData(UploadHandle uploadHandle, ResourceHandle destResourceHandle, D3D12_SUBRESOURCE_DATA *subresData, UINT numSubresources, UINT startSubresourceIndex) {
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

	return (bool)UpdateSubresources(cmdLists[uploadHandle].getComPtr().Get(), destResource, stageResource, 0, startSubresourceIndex, numSubresources, subresData);
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

ResourceHandle ResourceManager::registerResourceImpl(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, D3D12_RESOURCE_STATES state) {
	ResourceHandle handle;
	auto lock = resourcesCS.lock();
	if (!resourcePool.empty()) {
		handle = resourcePool.front();
		resourcePool.pop();
		// No need to make a new 
		resources[handle].res = resourcePtr;
		resources[handle].subresStates = SubresStates{ subresourcesCount, state };
	} else {
		resources.push_back(Resource{ resourcePtr, SubresStates(subresourcesCount, state), {} });
		handle = resources.size() - 1;
		if (numThreads > 1) {
			resources[handle].cs.init();
		}
	}

	return handle;
}

#ifdef D3D12_DEBUG
ResourceHandle ResourceManager::registerResource(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, D3D12_RESOURCE_STATES state, ResourceType type) {
	ResourceHandle handle = registerResourceImpl(resourcePtr, subresourcesCount, state);
	resources[handle].type = type;
	return handle;
}
#else
ResourceHandle ResourceManager::registerResource(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, D3D12_RESOURCE_STATES state) {
	return registerResourceImpl(resourcePtr, subresourcesCount, state);
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

void ResourceManager::endFrame() {
	// Destroy any staging resources
	for (int i = 0; i < stagingBuffers.size(); ++i) {
		deregisterResource(stagingBuffers[i]);
	}
}

void ResourceManager::resetCommandLists() {
	if (cmdLists.empty()) {
		cmdLists.emplace_back();
		return;
	}

	// Leave the first element to be our invalid upload handle.
	cmdLists.erase(cmdLists.begin() + 1, cmdLists.end());
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
