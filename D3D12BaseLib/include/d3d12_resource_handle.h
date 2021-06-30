#pragma once

#include "d3d12_defines.h"

struct ID3D12Resource;

static const SizeType INVALID_RESOURCE_HANDLE = 0;
struct ResourceHandle {
	SizeType handle;

	ResourceHandle() = default;
	ResourceHandle(SizeType handle) : handle(handle) { }
	ResourceHandle operator=(const SizeType &handle) {
		this->handle = handle;
		return *this;
	}

	bool operator==(const SizeType &handle) const {
		return this->handle == handle;
	}

	bool operator!=(const SizeType &handle) const {
		return !(this->handle == handle);
	}

	bool operator<(const SizeType &handle) const {
		return this->handle < handle;
	}

	bool operator<=(const SizeType &handle) const {
		return this->handle < handle || this->handle == handle;
	}

	bool operator>=(const SizeType &handle) const {
		return !(this->handle < handle);
	}

	bool operator>(const SizeType &handle) const {
		return !(this->handle <= handle);
	}

	operator SizeType&() {
		return handle;
	}

	operator SizeType() const {
		return handle;
	}

	ID3D12Resource* get();
	ID3D12Resource* operator->();
};
