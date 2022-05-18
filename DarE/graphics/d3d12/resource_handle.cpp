#include "d3d12/resource_handle.h"

#include "d3d12/includes.h"
#include "d3d12/resource_manager.h"

namespace Dar {

ID3D12Resource *ResourceHandle::get() const {
	ResourceManager &resManager = getResourceManager();
	return resManager.getID3D12Resource(handle);
}

ID3D12Resource *ResourceHandle::operator->() {
	return get();
}

const ID3D12Resource *ResourceHandle::operator->() const {
	return get();
}

} // namespace Dar