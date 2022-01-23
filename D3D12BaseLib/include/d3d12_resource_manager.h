#pragma once

#include "d3d12_async.h"
#include "d3d12_command_queue.h"
#include "d3d12_resource_handle.h"

using SubresStates = Vector<D3D12_RESOURCE_STATES>;
using UploadHandle = SizeType;

#define INVALID_UPLOAD_HANDLE SizeType(-1)

enum class ResourceType : unsigned int {
	Invalid = 0,
	DataBuffer,
	StagingBuffer,
	TextureBuffer,
	RenderTargetView,
	DepthStencilBuffer,
};

struct ResourceInitData {
	struct TextureData {
		UINT width = 0;
		UINT height = 0;
		UINT mipLevels = 0;
		UINT samplesCount = 1;
		UINT samplesQuality = 0;
		DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
	};

	struct StagingData {
		ResourceHandle destResource;
		UINT numSubresources = 1;
		UINT firstSubresourceIndex = 0;
	};

	ResourceType type = ResourceType::DataBuffer;
	union {
		SizeType size; ///< Size used when DataBuffer is created
		TextureData textureData; ///< Used when creating Texture/DepthStencil Buffer
		StagingData stagingData; ///< Used when creating staging buffer
	};
	D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST; ///< Used only when creating Data/Texture Buffer resource
	WString name = L""; ///< Empty name will result in setting default name corresponding to the type of the resource
	D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;

	ResourceInitData(ResourceType type) : type(type), size(0) {
		switch (type) {
		case ResourceType::DataBuffer:
			size = 0;
			break;
		case ResourceType::RenderTargetView:
		case ResourceType::DepthStencilBuffer:
		case ResourceType::TextureBuffer:
			textureData = {};
			break;
		case ResourceType::StagingBuffer:
			stagingData = {};
			break;
		default:
			break;
		}
	}
};

struct ResourceManager {
	/// Creates a buffer of the specified in ResourceInitData type.
	/// Be careful with 
	ResourceHandle createBuffer(const ResourceInitData &initData);

	// TODO: see if this method could be made idempotent per thread.
	/// Creates a new command list for use in upload*Data methods.
	/// Intended usage is for this to be called once from each thread before uploading data.
	/// Could be called more than once in 1 thread but there is no point in creating additional command lists.
	/// @return upload handle used by the ResourceManager to identify which command list will be used for uploading
	UploadHandle beginNewUpload();

	/// Upload 1D data to the specified GPU resource. Implicitly creates a staging buffer which
	/// is destroyed after the actual upload happens via uploadBuffers().
	/// @param uploadHandle A handle to an upload command list internally used by the resource manager
	/// @param destResource A handle to the destination resource
	/// @param data Pointer to CPU memory holding the data we wish to upload
	/// @param size Size in bytes of the data we wish to upload
	bool uploadBufferData(UploadHandle uploadHandle, ResourceHandle destResource, const void *data, SizeType size);
	
	/// Upload 2D data to the specified GPU resource. Array data is not supported!
	/// Implicitly creates a staging buffer which
	/// is destroyed after the actual upload happens via uploadBuffers().
	/// @param uploadHandle A handle to an upload command list internally used by the resource manager
	/// @param destResource A handle to the destination resource
	/// @param subresData An array of numSubresources elements holding data for each subresource we wish to upload.
	/// @param numSubresources Number of subresources we wish to upload
	/// @param startSubresourceIndex Index of the first subresource we wish to upload.
	bool uploadTextureData(UploadHandle uploadHandle, ResourceHandle destResource, D3D12_SUBRESOURCE_DATA *subresData, UINT numSubresources, UINT startSubresourceIndex);
	
	/// NOTE!!! Not thread safe!
	/// Should be called after all calls to upload*Data to actually submit the data to the GPU.
	/// Submits all command lists created for each upload handle
	/// Currently waits for the copying to finish.
	bool uploadBuffers();

	/// Flushes any command lists's work. Same as uploadBuffers()
	bool flush();

	/// Get the number of subresources for a resource or 0 if it's not tracked.
	unsigned int getSubresourcesCount(ResourceHandle handle);

	/// Get global states the subresources of a resource 
	bool getLastGlobalState(ResourceHandle handle, SubresStates &outStates);

	/// Get global state for an individual subresource of a resource
	bool getLastGlobalStateForSubres(ResourceHandle handle, D3D12_RESOURCE_STATES &outState, const unsigned int subresIndex);

	/// Set the global state of a resource for all of its subresources
	bool setGlobalState(ResourceHandle handle, const D3D12_RESOURCE_STATES &state);

	/// Set the global state of a subresource for all of its subresources
	bool setGlobalStateForSubres(ResourceHandle handle, const D3D12_RESOURCE_STATES &state, const unsigned int subresIndex);

#ifdef D3D12_DEBUG
	ResourceHandle registerResource(ComPtr<ID3D12Resource> resource, UINT subresourcesCount, D3D12_RESOURCE_STATES state, ResourceType type);
#else
	ResourceHandle registerResource(ComPtr<ID3D12Resource> resource, UINT subresourcesCount, D3D12_RESOURCE_STATES state);
#endif

	/// Release resource's data. All work with the resource is expected to have completed.
	bool deregisterResource(ResourceHandle &handle);

	ID3D12Resource* getID3D12Resource(ResourceHandle handle);

	void endFrame();

private:
	ResourceManager() : copyQueue(D3D12_COMMAND_LIST_TYPE_COPY), numThreads(1) { }

	void resetCommandLists();

	ResourceHandle registerResourceImpl(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, D3D12_RESOURCE_STATES state);

#ifdef D3D12_DEBUG
	ResourceType getResourceType(ResourceHandle handle);
#endif // D3D12_DEBUG

	struct Resource {
		ComPtr<ID3D12Resource> res;
		SubresStates subresStates;
		CriticalSection cs;
#ifdef D3D12_DEBUG
		ResourceType type = ResourceType::Invalid;
#endif // D3D12_DEBUG
	};

	ComPtr<ID3D12Device8> device;
	CommandQueue copyQueue;
	Vector<CommandList> cmdLists;

	Vector<Resource> resources;
	Vector<ResourceHandle> stagingBuffers;
	Queue<ResourceHandle> resourcePool;

	CriticalSection resourcesCS;
	unsigned int numThreads;

	friend bool initResourceManager(ComPtr<ID3D12Device8> device, const unsigned int numThreads);
	friend void deinitResourceManager();
};

bool initResourceManager(ComPtr<ID3D12Device8> device, const unsigned int numThreads);
void deinitResourceManager();
ResourceManager& getResourceManager();
