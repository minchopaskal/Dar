#pragma once

#include "utils/defines.h"

struct ID3D12Resource;

#define INVALID_RESOURCE_HANDLE SizeType(-1)

namespace Dar {

struct ResourceHandle {
	SizeType handle;

	ResourceHandle() : handle(INVALID_RESOURCE_HANDLE) {};
	ResourceHandle(SizeType handle) : handle(handle) {}
	ResourceHandle operator=(const SizeType &h) {
		handle = h;
		return *this;
	}

	bool operator==(const SizeType &h) const {
		return handle == h;
	}

	bool operator!=(const SizeType &h) const {
		return !(handle == h);
	}

	bool operator<(const SizeType &h) const {
		return handle < h;
	}

	bool operator<=(const SizeType &h) const {
		return handle < h || handle == h;
	}

	bool operator>=(const SizeType &h) const {
		return !(handle < h);
	}

	bool operator>(const SizeType &h) const {
		return !(handle <= h);
	}

	operator SizeType &() {
		return handle;
	}

	operator SizeType() const {
		return handle;
	}

	ID3D12Resource *get() const;

	ID3D12Resource *operator->();

	const ID3D12Resource *operator->() const;
};

} // namespace Dar
