#include "d3d12/resource_manager.h"

#include "async/job_system.h"

#include "AgilitySDK/include/d3d12.h"

#include "d3dx12.h"

#include "utils/utils.h"

#define CHECK_RESOURCE_HANDLE(handle) \
do { \
	if (handle == INVALID_RESOURCE_HANDLE || handle >= resources.size() || resources[handle].res == nullptr) { \
		return 0; \
	} \
} while (false)

#define CHECK_HEAP_HANDLE(handle) \
do { \
	if (handle == INVALID_HEAP_HANDLE || handle >= heaps.size() || heaps[handle].heap == nullptr) { \
		return 0; \
	} \
} while (false)

namespace Dar {

String getResourceNameByType(ResourceType type) {
	switch (type) {
	case ResourceType::DataBuffer:
		return "DataBuffer";
	case ResourceType::StagingBuffer:
		return "StagingBuffer";
	case ResourceType::DepthStencilBuffer:
		return "DepthStencilBuffer";
	case ResourceType::TextureBuffer:
		return "TextureBuffer";
	default:
		return "UnknownTypeBuffer";
	}
}

ResourceManager::ResourceManager(int nt) : copyQueue(D3D12_COMMAND_LIST_TYPE_COPY) {
	numThreads = nt;

	if (numThreads > 1) {
		resourcesCS.init();
		copyQueueCS.init();
	}

	cmdLists.resize(numThreads);
	stagingBuffers.resize(numThreads);
	uploadContexts.resize(numThreads);
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
	}

	HRESULT err = device->CreateHeap(&heapDesc, IID_PPV_ARGS(heaps[handle].heap.GetAddressOf()));
	if (err != S_OK) {
		deallocateHeap(handle);
		return INVALID_HEAP_HANDLE;
	}

	heaps[handle].alignment = alignment;
	heaps[handle].size = size;
	heaps[handle].offset = 0;

	return handle;
}

Vector<D3D12_RESOURCE_DESC> getResourceDescriptorsFromResourceInitData(const Vector<ResourceInitData> &resInitDatas) {
	Vector<D3D12_RESOURCE_DESC> resDescs;
	resDescs.resize(resInitDatas.size());

	for (int i = 0; i < resInitDatas.size(); ++i) {
		resDescs[i] = resInitDatas[i].getResourceDescriptor();
	}

	return resDescs;
}

HeapHandle ResourceManager::createHeap(const Vector<ResourceInitData> &resInitDatas) {
	auto resDescriptors = getResourceDescriptorsFromResourceInitData(resInitDatas);
	D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, static_cast<UINT>(resDescriptors.size()), resDescriptors.data());
	return createHeap(static_cast<SizeType>(allocInfo.SizeInBytes), static_cast<HeapAlignmentType>(allocInfo.Alignment));
}

void ResourceManager::createHeap(const Vector<ResourceInitData> &resInitDatas, HeapHandle &handle) {
	auto resDescriptors = getResourceDescriptorsFromResourceInitData(resInitDatas);
	
	D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, static_cast<UINT>(resDescriptors.size()), resDescriptors.data());
	HeapAlignmentType alignment = static_cast<HeapAlignmentType>(allocInfo.Alignment);
	SizeType size = static_cast<SizeType>(allocInfo.SizeInBytes);
	bool canReuse =
		isValidHeapHandle(handle) &&
		(alignment == getHeapAlignment(handle)) &&
		(getHeapOffset(handle) + size <= getHeapSize(handle));

	if (!canReuse) {
		deallocateHeap(handle);
		handle = createHeap(size, alignment);
	}
}

void ResourceManager::createHeap(D3D12_RESOURCE_DESC* resDescriptors, UINT numResources, HeapHandle& handle) {
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
	heaps[handle].offset = 0;

	heapHandlesPool.push(handle);

	handle = INVALID_HEAP_HANDLE;

	return true;
}

ResourceHandle ResourceManager::createBuffer(ResourceInitData &initData) {
	ResourceType type = initData.type;

	if (type == ResourceType::StagingBuffer) {
		CHECK_RESOURCE_HANDLE(initData.stagingData.destResource);
	}

	D3D12_RESOURCE_STATES initialState = initData.state;
	D3D12_CLEAR_VALUE clearValue = {};
	if (static_cast<uint32_t>(type) >= static_cast<uint32_t>(ResourceType::TextureBuffer)) {
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
		dassert(false);
		return INVALID_RESOURCE_HANDLE;
	}

	D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, 1, &resDesc);
	SizeType size = allocInfo.SizeInBytes;

	bool placedResource = false;
	if (initData.heapHandle.has_value() && isValidHeapHandle(*initData.heapHandle)) { // We have a heap on which to create the placed resource
		auto& heap = heaps[*initData.heapHandle];
		SizeType alignment = static_cast<SizeType>(heap.alignment);
		if (heap.offset % alignment != 0) { // make sure to pass valid offset, so we don't crash
			heap.offset = (heap.offset + alignment - 1) / alignment;
			heap.offset *= alignment;
		}

		HRESULT res = device->CreatePlacedResource(
			heap.heap.Get(),
			heap.offset,
			&resDesc,
			initialState,
			clearValuePtr,
			IID_PPV_ARGS(resource.GetAddressOf())
		);

		if (res == S_OK) {
			placedResource = true;
			heap.offset += size;
		} else {
			auto err = GetLastError();
			LOG_FMT(Warning, "Failed to create placed resource with error: %lu! Falling back on creating a commited resource!\n", err);
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

	String name = initData.name.size() == 0 ? getResourceNameByType(type) : initData.name;
	resource->SetName(strToWStr(name).c_str());

	UINT numSubresources = 1;
	if (type == ResourceType::TextureBuffer && initData.textureData.mipLevels > 0) {
		numSubresources = initData.textureData.mipLevels;
	} else if (type == ResourceType::StagingBuffer) {
		numSubresources = initData.stagingData.numSubresources;
	}

#ifdef DAR_NDEBUG
	return registerResource(
		resource,
		numSubresources,
		size,
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
#endif // DAR_NDEBUG
}

UploadHandle ResourceManager::beginNewUpload() {
	const auto threadIdx = JobSystem::getCurrentThreadIndex();
	
	auto lock = copyQueueCS.lock();
	cmdLists[threadIdx].emplace_back(copyQueue.getCommandList());
	return cmdLists[threadIdx].size() - 1;
}

bool ResourceManager::uploadBufferData(UploadHandle uploadHandle, ResourceHandle destResourceHandle, const void *data, SizeType size) {
	CHECK_RESOURCE_HANDLE(destResourceHandle);

	ResourceInitData resData(ResourceType::StagingBuffer);
	resData.stagingData.destResource = destResourceHandle;

	ResourceHandle stagingBufferHandle = createBuffer(resData);
	if (stagingBufferHandle == INVALID_RESOURCE_HANDLE) {
		return 0;
	}

#ifdef DAR_DEBUG
	SubresStates states;
	getLastGlobalState(destResourceHandle, states);
	dassert(states.size() == 1);
#endif // DAR_DEBUG

	auto stageResource = getID3D12Resource(stagingBufferHandle);

	UINT8 *mappedBuffer = nullptr;
	D3D12_RANGE range = { 0, 0 };
	stageResource->Map(0, &range, reinterpret_cast<void **>(&mappedBuffer));
	memcpy(mappedBuffer, data, size);
	stageResource->Unmap(0, &range);

	const auto threadIdx = JobSystem::getCurrentThreadIndex();
	registerStagingBuffer(uploadHandle, stagingBufferHandle);
	cmdLists[threadIdx][uploadHandle].copyBufferRegion(destResourceHandle, stagingBufferHandle, size);

	return true;
}

UINT64 ResourceManager::uploadTextureData(UploadHandle uploadHandle, ResourceHandle destResourceHandle, D3D12_SUBRESOURCE_DATA *subresData, UINT numSubresources, UINT startSubresourceIndex) {
	CHECK_RESOURCE_HANDLE(destResourceHandle);

	ResourceInitData resData(ResourceType::StagingBuffer);
	resData.stagingData.destResource = destResourceHandle;
	resData.stagingData.firstSubresourceIndex = startSubresourceIndex;
	resData.stagingData.numSubresources = numSubresources;

	ResourceHandle stagingBufferHandle = createBuffer(resData);
	if (stagingBufferHandle == INVALID_RESOURCE_HANDLE) {
		return 0;
	}

	ID3D12Resource *destResource = getID3D12Resource(destResourceHandle);
	ID3D12Resource *stageResource = getID3D12Resource(stagingBufferHandle);

#ifdef DAR_DEBUG
	SubresStates states;
	getLastGlobalState(destResourceHandle, states);
	dassert(states.size() >= numSubresources + startSubresourceIndex);
#endif // DAR_DEBUG

	const auto threadIdx = JobSystem::getCurrentThreadIndex();
	registerStagingBuffer(uploadHandle, stagingBufferHandle);
	return UpdateSubresources(cmdLists[threadIdx][uploadHandle].get(), destResource, stageResource, 0, startSubresourceIndex, numSubresources, subresData);
}

bool ResourceManager::uploadBuffers() {
	const auto handle = uploadBuffersAsync();
	const bool result = waitUploadFence(handle);

	return result;
}

UploadContextHandle ResourceManager::uploadBuffersAsync() {
	const auto threadIdx = JobSystem::getCurrentThreadIndex();

	Vector<ResourceHandle> stagingBuffersToRelease;
	auto &stagingBuffs = stagingBuffers[threadIdx];
	auto &threadCmdLists = cmdLists[threadIdx];

	for (int i = 0; i < threadCmdLists.size(); ++i) {
		if (auto it = stagingBuffs.find(i); it != stagingBuffs.end()) {
			for (auto handle : it->second) {
				stagingBuffersToRelease.push_back(handle);
			}
		}
		stagingBuffs.erase(i);
	}

	// TODO: see if it's reasonable to have multiple copy queues
	// per thread. Guess is it will not be much quicker as there wouldn't
	// be so many copy engines on the gpu(if any at all).
	auto lock = copyQueueCS.lock();

	for (int i = 0; i < threadCmdLists.size(); ++i) {
		copyQueue.addCommandListForExecution(std::move(threadCmdLists[i]));

	}
	resetCommandLists();

	FenceValue fence = copyQueue.executeCommandLists();

	UploadContext uploadCtx = {};
	uploadCtx.fence = fence;
	uploadCtx.buffersToRelease = stagingBuffersToRelease;

	return uploadContexts[threadIdx].push(uploadCtx);
}

bool ResourceManager::waitUploadFence(UploadContextHandle handle) {
	const auto threadIdx = JobSystem::getCurrentThreadIndex();

	auto &uploadCtx = uploadContexts[threadIdx].at(handle);
	if (!uploadCtx.has_value()) {
		return false;
	}

	copyQueue.waitForFenceValue(uploadCtx->fence);
	for (auto bufHandle : uploadCtx->buffersToRelease) {
		deregisterResource(bufHandle);
	}

	dassert(uploadContexts[threadIdx].release(handle));

	return true;
}

bool ResourceManager::flush() {
	// TODO: Currently doesn't work as intended, since it depends on the thread it was called on.
	return uploadBuffers();
}

unsigned int ResourceManager::getSubresourcesCount(ResourceHandle handle) {
	CHECK_RESOURCE_HANDLE(handle);

	return (unsigned int)(resources[handle].subresStates.size());
}

bool ResourceManager::getLastGlobalState(ResourceHandle handle, SubresStates &outStates) {
	CHECK_RESOURCE_HANDLE(handle);

	auto lock = resources[handle].cs.lock();
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
	auto lock = resources[handle].cs.lock();
	SubresStates &states = resources[handle].subresStates;
	if (subresIndex < 0 || states.size() <= subresIndex) {
		return false;
	}

	outState = states[subresIndex];
	return true;
}

bool ResourceManager::setGlobalState(ResourceHandle handle, const D3D12_RESOURCE_STATES &state) {
	CHECK_RESOURCE_HANDLE(handle);

	auto lock = resources[handle].cs.lock();
	SubresStates &states = resources[handle].subresStates;
	for (int i = 0; i < states.size(); ++i) {
		states[i] = state;
	}

	return true;
}

bool ResourceManager::setGlobalStateForSubres(ResourceHandle handle, const D3D12_RESOURCE_STATES &state, const unsigned int subresIndex) {
	CHECK_RESOURCE_HANDLE(handle);

	auto lock = resources[handle].cs.lock();
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

void ResourceManager::registerStagingBuffer(UploadHandle uploadHandle, ResourceHandle resourceHandle) {
	const auto threadIdx = JobSystem::getCurrentThreadIndex();

	auto& bufs = stagingBuffers[threadIdx];
	if (auto it = bufs.find(uploadHandle); it == bufs.end()) {
		bufs.insert({ uploadHandle, Vector<ResourceHandle>{} });
	}

	bufs[uploadHandle].push_back(resourceHandle);
}

#ifdef DAR_DEBUG
ResourceHandle ResourceManager::registerResource(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state, ResourceType type) {
	ResourceHandle handle = registerResourceImpl(resourcePtr, subresourcesCount, size, state);
	resources[handle].type = type;
	return handle;
}
#else
ResourceHandle ResourceManager::registerResource(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state) {
	return registerResourceImpl(resourcePtr, subresourcesCount, size, state);
}
#endif // DAR_DEBUG

bool ResourceManager::deregisterResource(ResourceHandle &handle) {
	CHECK_RESOURCE_HANDLE(handle);

	{
		auto lock = resourcesCS.lock();
#pragma warning(suppress: 4189)
		unsigned long refCount = resources[handle].res.Reset();

#ifdef DAR_DEBUG

		ResourceType type = getResourceType(handle);
		dassert(type != ResourceType::Invalid);
		if (type == ResourceType::StagingBuffer) {
			dassert(refCount == 0);
		}
#endif // DAR_DEBUG


		resourcePool.push(handle);
	}

	handle = INVALID_RESOURCE_HANDLE;
	return true;
}

ID3D12Resource *ResourceManager::getID3D12Resource(ResourceHandle handle) const {
	CHECK_RESOURCE_HANDLE(handle);

	return resources[handle].res.Get();
}

ID3D12Heap *ResourceManager::getID3D12Heap(HeapHandle handle) const {
	CHECK_HEAP_HANDLE(handle);
	return heaps[handle].heap.Get();
}

HeapAlignmentType ResourceManager::getHeapAlignment(HeapHandle handle) const {
	if (!isValidHeapHandle(handle)) {
		return HeapAlignmentType::Invalid;
	}
	return heaps[handle].alignment;
}

SizeType ResourceManager::getHeapSize(HeapHandle handle) const {
	CHECK_HEAP_HANDLE(handle);
	return heaps[handle].size;
}

SizeType ResourceManager::getHeapOffset(HeapHandle handle) const {
	CHECK_HEAP_HANDLE(handle);
	return heaps[handle].offset;
}

void ResourceManager::endFrame() {
}

SizeType ResourceManager::getResourceSize(ResourceHandle handle) const {
	if (handle == INVALID_RESOURCE_HANDLE) {
		return 0;
	}

	return resources[handle].size;
}

ResourceManager::~ResourceManager() {
	for (int i = 0; i < resources.size(); ++i) {
#pragma warning(suppress: 4189)
		const unsigned long refCount = resources[i].res.Reset();

#ifdef DAR_DEBUG
		// The resource manager has ownership of all the resources besides
		// the RenderTargetBuffers which are created by the swapchain.
		// Here we make sure we didn't leak any of the other resources
		// to other places. That way we know exactly how much they live.
		if (resources[i].type != ResourceType::RenderTargetBuffer) {
			dassert(refCount == 0);
		}
#endif // DAR_DEBUG
	}
}

void ResourceManager::resetCommandLists() {
	const auto threadIdx = JobSystem::getCurrentThreadIndex();

	cmdLists[threadIdx].clear();
}

bool ResourceManager::isValidHeapHandle(HeapHandle handle) const {
	return !(handle == INVALID_HEAP_HANDLE || handle >= heaps.size() || heaps[handle].heap == nullptr);
}

#ifdef DAR_DEBUG
ResourceType ResourceManager::getResourceType(ResourceHandle handle) {
	return resources[handle].type;
}
#endif // DAR_DEBUG

ResourceManager *g_ResourceManager = nullptr;

bool initResourceManager(ComPtr<ID3D12Device> device, const unsigned int nt) {
	deinitResourceManager();

	g_ResourceManager = new ResourceManager(nt);
	g_ResourceManager->device = device;
	g_ResourceManager->copyQueue.init(device);

	return true;
}

void deinitResourceManager() {
	if (g_ResourceManager) {
		delete g_ResourceManager;
		g_ResourceManager = nullptr;
	}
}

ResourceManager &getResourceManager() {
	if (g_ResourceManager == nullptr) {
		dassert(false);
	}

	return *g_ResourceManager;
}

D3D12_RESOURCE_DESC ResourceInitData::getResourceDescriptor() const {
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
			textureData.depth,
			static_cast<UINT16>(textureData.mipLevels),
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

void ResourceInitData::init(ResourceType t) {
	type = t;
	switch (type) {
	case ResourceType::DataBuffer:
		size = 0;
		break;
	case ResourceType::TextureBuffer:
	case ResourceType::RenderTargetBuffer:
		textureData = {};
		textureData.clearValue.color[0] = 0.f;
		textureData.clearValue.color[1] = 0.f;
		textureData.clearValue.color[2] = 0.f;
		textureData.clearValue.color[3] = 1.f;
		break;
	case ResourceType::DepthStencilBuffer:
		textureData = {};
		textureData.clearValue.depthStencil.depth = 1.f;
		textureData.clearValue.depthStencil.stencil = 0;
		break;
	case ResourceType::StagingBuffer:
		stagingData = {};
		break;
	default:
		break;
	}
}

} // namespace Dar
