#pragma once

#include "d3d12_defines.h"

namespace Dar {

struct Random {
	Random() {
		std::random_device rd{};
		generator.seed(rd());
	}

	template <class T>
	std::enable_if_t<std::is_floating_point_v<T>, T> generateFlt(T min, T max) {
		std::uniform_real_distribution<double> uniformDist{ double(min), double(max) };
		return static_cast<T>(uniformDist(generator));
	}

	template <class T>
	std::enable_if_t<std::is_integral_v<T>, T> generateInt(T min, T max) {
		std::uniform_int_distribution<SizeType> uniformDist{ static_cast<SizeType>(min), static_cast<SizeType>(max) };
		return static_cast<T>(uniformDist(generator));
	}

	template <class T>
	std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, T> generateIntSigned(T min, T max) {
		std::uniform_int_distribution<long long> uniformDist{ static_cast<long long>(min),  static_cast<long long>(max) };
		return static_cast<T>(uniformDist(generator));
	}

private:
	std::mt19937_64 generator;
};

} // namespace Dar
