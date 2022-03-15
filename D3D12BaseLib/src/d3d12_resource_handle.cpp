#include "d3d12_resource_handle.h"

#include "d3d12_includes.h"
#include "d3d12_resource_manager.h"

ID3D12Resource* ResourceHandle::get() {
	ResourceManager &resManager = getResourceManager();
	return resManager.getID3D12Resource(handle);
}

ID3D12Resource* ResourceHandle::operator->() {
	return get();
}

const ID3D12Resource *ResourceHandle::get() const {
	ResourceManager &resManager = getResourceManager();
	return resManager.getID3D12Resource(handle);
}

const ID3D12Resource *ResourceHandle::operator->() const {
	return get();
}

