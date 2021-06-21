#pragma once

#include "d3d12_defines.h"

struct ID3D12Resource;
enum D3D12_RESOURCE_STATES : int;

namespace ResourceTracker {

bool init(const unsigned int numThreads);
void deinit();

bool registerResource(ID3D12Resource *resource, const unsigned int numSubresources);
bool registerResource(ID3D12Resource *resource, const Vector<D3D12_RESOURCE_STATES> &states);
void unregisterResource(ID3D12Resource *resource);

/// Get the number of subresources for a resource or 0 if it's not tracked.
unsigned int getSubresourcesCount(ID3D12Resource *resource);

/// Get global states the subresources of a resource 
bool getLastGlobalState(ID3D12Resource *resource, Vector<D3D12_RESOURCE_STATES> &outStates);

/// Get global state for an individual subresource of a resource
bool getLastGlobalStateForSubres(ID3D12Resource *resource, D3D12_RESOURCE_STATES &outState, const unsigned int subresIndex);

/// Set the global state of a resource for all of its subresources
bool setGlobalState(ID3D12Resource *resource, const D3D12_RESOURCE_STATES &state);

/// Set the global state of a subresource for all of its subresources
bool setGlobalStateForSubres(ID3D12Resource *resource, const D3D12_RESOURCE_STATES &state, const unsigned int subresIndex);

} // namespace ResourceTracker