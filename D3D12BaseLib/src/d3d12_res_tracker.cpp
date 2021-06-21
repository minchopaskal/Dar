#include "d3d12_res_tracker.h"

#include "AgilitySDK/include/d3d12.h"
#include "d3d12_defines.h"
#include "d3d12_async.h"

#include <unordered_map>
#include <vector>

using SubresStates = Vector<D3D12_RESOURCE_STATES>;
struct ResourceData {
	SubresStates states;
	CriticalSection cs;
};

/// Last global state saved for each tracked resource. Also saves N critical sections for each resource, where N is the number of threads.
using GlobalStatesMap = Map<ID3D12Resource*, ResourceData>;
GlobalStatesMap globalStates;

CriticalSection globalStatesCS;
int numThreads = 0;

bool ResourceTracker::init(const unsigned int nt) {
	numThreads = nt;
	if (nt > 1) {
		return globalStatesCS.initialize();
	}
	return true;
}

void ResourceTracker::deinit() { }

bool ResourceTracker::registerResource(ID3D12Resource *resource, const unsigned int numSubresources) {
	ResourceTracker::registerResource(resource, Vector<D3D12_RESOURCE_STATES>(numSubresources, D3D12_RESOURCE_STATE_COMMON));
}

bool ResourceTracker::registerResource(ID3D12Resource *resource, const Vector<D3D12_RESOURCE_STATES> &states) {
	if (!resource || globalStates.find(resource) != globalStates.end()) {
		return false;
	}

	if (states.size() == 0) {
		return false;
	}

	ResourceData data;
	data.states = SubresStates(states);
	if (numThreads > 1) {
		if (!data.cs.initialize()) {
			return false;
		}
	}

	auto lock = globalStatesCS.lock();
	globalStates[resource] = data;

	return true;
}

void ResourceTracker::unregisterResource(ID3D12Resource *resource) {
	if (!resource || globalStates.find(resource) == globalStates.end()) {
		return;
	}

	auto lock = globalStatesCS.lock();
	globalStates.erase(resource);
}

unsigned int ResourceTracker::getSubresourcesCount(ID3D12Resource * resource) {
	if (!resource || globalStates.find(resource) == globalStates.end()) {
		return 0;
	}

	return (unsigned int)(globalStates[resource].states.size());
}

bool ResourceTracker::getLastGlobalState(ID3D12Resource *resource, Vector<D3D12_RESOURCE_STATES> &outStates) {
	if (!resource || globalStates.find(resource) == globalStates.end()) {
		return false;
	}

	CriticalSectionLock lock = globalStates[resource].cs.lock();
	SubresStates &states = globalStates[resource].states;
	outStates.resize(states.size());
	for (int i = 0; i < states.size(); ++i) {
		outStates[i] = states[i];
	}

	return true;
}

bool ResourceTracker::getLastGlobalStateForSubres(ID3D12Resource *resource, D3D12_RESOURCE_STATES &outState, const unsigned int subresIndex) {
	if (!resource || globalStates.find(resource) == globalStates.end()) {
		return false;
	}

	// TODO: Experiment with locking individual subresources
	CriticalSectionLock lock = globalStates[resource].cs.lock();
	SubresStates &states = globalStates[resource].states;
	if (subresIndex == 0 || states.size() <= subresIndex) {
		return false;
	}

	outState = states[subresIndex];
	return true;
}

bool ResourceTracker::setGlobalState(ID3D12Resource *resource, const D3D12_RESOURCE_STATES &state) {
	if (!resource || globalStates.find(resource) == globalStates.end()) {
		return false;
	}
	
	CriticalSectionLock lock = globalStates[resource].cs.lock();
	SubresStates &states = globalStates[resource].states;
	for (int i = 0; i < states.size(); ++i) {
		states[i] = state;
	}

	return true;
}

bool ResourceTracker::setGlobalStateForSubres(ID3D12Resource *resource, const D3D12_RESOURCE_STATES &state, const unsigned int subresIndex) {
	if (!resource || globalStates.find(resource) == globalStates.end()) {
		return false;
	}

	CriticalSectionLock lock = globalStates[resource].cs.lock();
	SubresStates &states = globalStates[resource].states;
	if (subresIndex == 0 || states.size() <= subresIndex) {
		return false;
	}
	states[subresIndex] = state;

	return true;
}

