#pragma once

#include "d3d12/descriptor_heap.h"
#include "d3d12/includes.h"
#include "d3d12/resource_handle.h"
#include "d3d12/texture_res.h"
#include "utils/utils.h"

namespace Dar {

struct DataBufferResource {
	/// Intialize or resize the data buffer.
	/// @param elemSize Size of the elements in the buffer to be allocated.
	/// @param numElems Number of elements in the buffer to be allocated.
	/// @param heapInfo Optional pointer to a HeapInfo object describing the heap on which we want the allocation to happen.
	/// @return true on success, false otherwise.
	bool init(SizeType elemSize, SizeType numElems, Optional<HeapHandle> heapHandle = std::nullopt);

	/// Upload CPU data to the buffer.
	/// @uploadHandle Upload handle object created by the resource manager. Used when batch uploading.
	/// @data Data to be uploaded.
	/// @return true on success, false otherwise.
	bool upload(UploadHandle uploadHandle, const void *data);

	ResourceHandle getHandle() const {
		return handle;
	}

	SizeType getSize() const {
		return size;
	}

	SizeType getNumElements() const {
		return size / elementSize;
	}

	SizeType getElementSize() const {
		return elementSize;
	}

	void setName(const String &name) {
		handle.get()->SetName(strToWStr(name).c_str());
	}

	void deinit();

private:
	ResourceHandle handle = INVALID_RESOURCE_HANDLE;
	SizeType size = 0;
	SizeType elementSize = 0;

	friend class ReadWriteBufferResource;
};

} // namespace Dar