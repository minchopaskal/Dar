#pragma once

#include "utils/defines.h"

struct ID3D12Resource;

#define INVALID_RESOURCE_HANDLE SizeType(-1)

struct ResourceHandle {
	SizeType handle;

	ResourceHandle() : handle(INVALID_RESOURCE_HANDLE) { };
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
	const ID3D12Resource* get() const;

	ID3D12Resource* operator->();
	
	const ID3D12Resource* operator->() const;
};
