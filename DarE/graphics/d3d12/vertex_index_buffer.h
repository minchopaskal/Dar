#pragma once

#include "d3d12/includes.h"
#include "d3d12/resource_manager.h"

namespace Dar {

enum class BufferType {
	Vertex,
	Index,

	Count
};

struct VertexIndexBufferDesc {
	const void *data = nullptr;
	WString name = L"GenericVertexIndexBuffer";
	UINT size = 0;
	union {
		int vertexBufferStride = 0;
		DXGI_FORMAT indexBufferFormat;
	};
};

template <BufferType T>
struct BufferBase;

template <>
struct BufferBase<BufferType::Vertex> {
	BufferBase() : bufferView{} {}

protected:
	void prepareBufferSpecificData(VertexIndexBufferDesc &desc) {
		bufferView.StrideInBytes = desc.vertexBufferStride;
	}

public:
	D3D12_VERTEX_BUFFER_VIEW bufferView;
};

template <>
struct BufferBase<BufferType::Index> {
	BufferBase() : bufferView{} {}

protected:
	void prepareBufferSpecificData(VertexIndexBufferDesc &desc) {
		bufferView.Format = desc.indexBufferFormat;
	}

public:
	D3D12_INDEX_BUFFER_VIEW bufferView;
};

template <BufferType T>
struct Buffer : BufferBase<T> {
	Buffer() : bufferHandle(INVALID_RESOURCE_HANDLE) {}

	/// Initialize and upload the vertex/index buffer.
	/// @param desc Vertex/index buffer description
	/// @param uploadHandle The upload handle used for recording the host->gpu copy
	/// @return True if initialization of the buffer and the upload command recording were successful.
	bool init(VertexIndexBufferDesc &desc, UploadHandle uploadHandle) {
		deinit();

		ResourceManager &resManager = getResourceManager();

		ResourceInitData dataDesc(ResourceType::DataBuffer);
		dataDesc.size = desc.size;
		dataDesc.name = desc.name;
		bufferHandle = resManager.createBuffer(dataDesc);
		if (bufferHandle == INVALID_RESOURCE_HANDLE) {
			return false;
		}

		this->bufferView.BufferLocation = bufferHandle->GetGPUVirtualAddress();
		this->bufferView.SizeInBytes = desc.size;
		this->prepareBufferSpecificData(desc);

		return resManager.uploadBufferData(uploadHandle, bufferHandle, desc.data, desc.size);
	}

private:
	void deinit() {
		ResourceManager &resManager = getResourceManager();
		resManager.deregisterResource(bufferHandle);
		this->bufferView = {};
	}

public:
	ResourceHandle bufferHandle;
};

using VertexBuffer = Buffer<BufferType::Vertex>;
using IndexBuffer = Buffer<BufferType::Index>;

} // namespace Dar