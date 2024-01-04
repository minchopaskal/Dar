#pragma once

#include "async/async.h"
#include "utils/defines.h"

namespace Dar {

using PooledIndex = SizeType;
#define INVALID_POOLED_INDEX SizeType(-1)

template <class T>
class PooledVector {
public:
	PooledVector() = default;

	PooledIndex push(const T &v) {
		if (freeIndices.empty()) {
			arr.push_back(v);
			return arr.size() - 1;
		}

		auto idx = freeIndices.front();
		freeIndices.pop();
		
		arr[idx] = v;
		
		return idx;
	}

	bool release(PooledIndex &index) {
		auto idx = index;
		index = INVALID_POOLED_INDEX;

		if (idx == INVALID_POOLED_INDEX) {
			return false;
		}

		if (idx >= arr.size()) {
			return false;
		}

		if (!arr[idx].has_value()) {
			return false;
		}

		arr[idx] = std::nullopt;
		freeIndices.push(idx);
		return true;
	}

	const std::optional<T>& at(PooledIndex idx) const {
		if (idx == INVALID_POOLED_INDEX || idx >= arr.size()) {
			return std::nullopt;
		}

		return arr[idx];
	}

private:
	Vector<std::optional<T>> arr;
	Queue<PooledIndex> freeIndices;
};

}