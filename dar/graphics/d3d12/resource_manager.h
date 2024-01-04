#pragma once

#include "async/async.h"
#include "d3d12/command_queue.h"
#include "d3d12/resource_handle.h"

#include "utils/pooled_vector.h"

namespace Dar {

using SubresStates = Vector<D3D12_RESOURCE_STATES>;
using UploadHandle = SizeType;
using HeapHandle = SizeType;
using UploadContextHandle = PooledIndex;

#define INVALID_UPLOAD_HANDLE SizeType(-1)
#define INVALID_HEAP_HANDLE SizeType(-1)
#define INVALID_UPLOAD_FENCE INVALID_POOLED_INDEX

enum class ResourceType : unsigned int {
	Invalid = 0,
	DataBuffer = 1,
	StagingBuffer = 2,
	
	TextureBuffer,
	RenderTargetBuffer,
	DepthStencilBuffer,
};

enum class HeapAlignmentType : SizeType {
	Default = 65536, // 64KB
	Small = 4069, // 4KB
	MSAA = 4194304, // 4 MB

	Invalid = 0
};

struct TextureInitData {
	UINT width = 0;
	UINT height = 0;
	UINT16 depth = 1;
	UINT mipLevels = 1;
	UINT samplesCount = 1;
	UINT samplesQuality = 0;
	DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
	union {
		FLOAT color[4];
		struct {
			FLOAT depth;
			UINT8 stencil;
		} depthStencil;
	} clearValue;
};

struct ResourceInitData {
	struct StagingInitData {
		ResourceHandle destResource = INVALID_RESOURCE_HANDLE;
		UINT numSubresources = 1;
		UINT firstSubresourceIndex = 0;
	};

	Optional<HeapHandle> heapHandle = std::nullopt;
	ResourceType type = ResourceType::DataBuffer;
	union {
		SizeType size; ///< Used when DataBuffer is created
		TextureInitData textureData; ///< Used when creating Texture/DepthStencil Buffer
		StagingInitData stagingData; ///< Used when creating staging buffer
	};
	D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON; ///< Used only when creating Data/Texture Buffer resource
	String name = ""; ///< Empty name will result in setting default name corresponding to the type of the resource
	D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;

	ResourceInitData() : type(ResourceType::Invalid), size(0) {}

	ResourceInitData(ResourceType type) : type(type), size(0) {
		init(type);
	}

	D3D12_RESOURCE_DESC getResourceDescriptor() const;

	void init(ResourceType type);
};

struct ResourceManager {
	CommandQueue &getCopyQueue() {
		return copyQueue;
	}

	const CommandQueue &getCopyQueue() const {
		return copyQueue;
	}

	/// Create a heap which can be used for placed resources.
	/// Use createBuffer to create a resource on the heap by specifying
	/// the ResourceInitData::heapInfo member.
	/// @param size Size of the heap we wish to create
	/// @param alignment Alignment
	/// @return Handle to the created heap.
	HeapHandle createHeap(SizeType size, HeapAlignmentType alignment);

	/// \see createHeap(SizeType). Calculates size of the heap by the given arguments.
	/// @param resInitDatas Array of resource descriptors.
	/// @return Handle to the created heap.
	HeapHandle createHeap(const Vector<ResourceInitData> &resInitDatas);

	/// \see createHeap(SizeType). Calculates size of the heap by the given arguments.
	/// @param resInitDatas Array of resource descriptors.
	/// @param handle Handle to the created heap. If handle points to an existing heap, try to reuse it if possible.
	void createHeap(const Vector<ResourceInitData> &resInitDatas, HeapHandle &handle);

	/// \see createHeap(SizeType). Calculates size of the heap by the given arguments.
/// @param resInitDatas Array of resource descriptors.
/// @param handle Handle to the created heap. If handle points to an existing heap, try to reuse it if possible.
	void createHeap(D3D12_RESOURCE_DESC *resInitDatas, UINT count, HeapHandle& handle);

	/// Release the memory of a created heap.
	/// @param handle Reference to the handle of the heap we wish to deallocate.
	/// @return true on success, false otherwise.
	bool deallocateHeap(HeapHandle &handle);

	/// Creates a buffer of the specified in ResourceInitData type.
	/// Resources can be commited or placed, depending on whether ResourceInitData::heapInfo is initialized.
	/// @param initData Resource initialization data.
	/// @return Handle to the resource, which is managed by the resource manager.
	ResourceHandle createBuffer(ResourceInitData &initData);

	/// @NOTE beginNewUpload, upload*Data and uploadBuffers are per-thread-safe
	///       meaning it'll be okay if you beginUpload in one thread and all subsequent upload*Data
	///       calls with the returned uploadHandle are done inside the same thread.
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
	/// @return Size of the uploaded texture. Useful when uploading to a heap, so we know how much the next resource will be offset in the heap.
	UINT64 uploadTextureData(UploadHandle uploadHandle, ResourceHandle destResource, D3D12_SUBRESOURCE_DATA *subresData, UINT numSubresources, UINT startSubresourceIndex);

	/// TODO: thread-safe, but may be SLOW if many threads are trying to beginNewUpload/upload*Data
	///       during an uploadBuffers() call.
	/// @note Thread-safe
	/// Should be called after all calls to upload*Data to actually submit the data to the GPU.
	/// Submits all command lists created for each upload handle
	/// Waits for the copying to finish.
	bool uploadBuffers();

	/// Same as uploadBuffers but doesn't wait for copying to finish.
	/// One must wait on the returned UploadContextHandle in order to wait for the upload to finish.
	/// @return an upload context handle that is an opaque to the resource manager user.
	UploadContextHandle uploadBuffersAsync();

	/// @brief Wait on an upload fence
	/// @return false if the fence is invalid
	bool waitUploadFence(UploadContextHandle fence);

	// TODO: this method makes the queue wait on the upload fence withotut blocking.
	// Make sure to save the upload ctx somewhere so the staging buffers are freed only after the frame has ended.
	bool waitUploadFence(CommandQueue &queue, UploadContextHandle fence) { return false; }

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

#ifdef DAR_DEBUG
	ResourceHandle registerResource(ComPtr<ID3D12Resource> resource, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state, ResourceType type);
#else
	ResourceHandle registerResource(ComPtr<ID3D12Resource> resource, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state);
#endif

	/// Release resource's data. All work with the resource is expected to have completed.
	bool deregisterResource(ResourceHandle &handle);

	ID3D12Resource *getID3D12Resource(ResourceHandle handle) const;
	SizeType getResourceSize(ResourceHandle handle) const;

	ID3D12Heap *getID3D12Heap(HeapHandle handle) const;
	HeapAlignmentType getHeapAlignment(HeapHandle handle) const;
	SizeType getHeapSize(HeapHandle handle) const;
	SizeType getHeapOffset(HeapHandle handle) const;

	void endFrame();

#ifdef DAR_DEBUG
	[[nodiscard]] ResourceType getResourceType(ResourceHandle handle);
#endif // DAR_DEBUG

private:
	ResourceManager(int nt);
	~ResourceManager();

	void resetCommandLists();

	bool isValidHeapHandle(HeapHandle handle) const;

	ResourceHandle registerResourceImpl(ComPtr<ID3D12Resource> resourcePtr, UINT subresourcesCount, SizeType size, D3D12_RESOURCE_STATES state);

	/// @NOTE NOT Thread-safe
	void registerStagingBuffer(UploadHandle uploadHandle, ResourceHandle resourceHandle);

	struct Resource {
		ComPtr<ID3D12Resource> res;
		SubresStates subresStates;
		CriticalSection cs;
		SizeType size;
#ifdef DAR_DEBUG
		ResourceType type = ResourceType::Invalid;
#endif // DAR_DEBUG
	};

	struct Heap {
		ComPtr<ID3D12Heap> heap = nullptr;
		HeapAlignmentType alignment = HeapAlignmentType::Invalid;
		SizeType size = 0;
		SizeType offset = 0;
	};

	struct UploadContext {
		FenceValue fence;
		Vector<ResourceHandle> buffersToRelease;
	};

	Vector<PooledVector<UploadContext>> uploadContexts;

	ComPtr<ID3D12Device> device;
	CommandQueue copyQueue;

	Vector<Vector<CommandList>> cmdLists;
	/// Map each staging buffer to its corresponding uploadHandle
	/// At the end of the upload all of its staging buffers will be
	/// released
	Vector<Map<UploadHandle, Vector<ResourceHandle>>> stagingBuffers;

	Vector<Resource> resources;
	Queue<ResourceHandle> resourcePool;

	Vector<Heap> heaps;
	Queue<HeapHandle> heapHandlesPool;

	CriticalSection resourcesCS;
	CriticalSection copyQueueCS;;
	unsigned int numThreads;

	friend bool initResourceManager(ComPtr<ID3D12Device> device, const unsigned int numThreads);
	friend void deinitResourceManager();
};

bool initResourceManager(ComPtr<ID3D12Device> device, const unsigned int numThreads);
void deinitResourceManager();
ResourceManager &getResourceManager();

} // namespace Dar