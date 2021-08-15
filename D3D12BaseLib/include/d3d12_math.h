#pragma once

#include <cmath>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#ifdef __CUDACC__
__device__ {
#endif // __CUDACC__

namespace dmath {

template <class T>
T radians(T degrees) {
	return degrees * static_cast< T >(0.01745329251994329576923690768489);
}

template <class T>
T degrees(T radians) {
	return radians * static_cast< T >(57.295779513082320876798154814105);
}

template <class T>
T max(T v1, T v2) {
	return v1 > v2 ? v1 : v2;
}

template <class T>
T min(T v1, T v2) {
	return v1 < v2 ? v1 : v2;
}

#pragma pack(push, 1)
namespace /*Packed*/ {

template <typename T>
struct Vec2t {
	union {
		struct {
			T x, y;
		};
		struct {
			T u, v;
		};
		T data[2];
	};

	Vec2t() {
		memset(this, 0, 2 * sizeof(T));
	}

	Vec2t(T x) : x(x), y(x) { }

	Vec2t(T x, T y) : x(x), y(y) { }

	Vec2t(const Vec2t &other) : x(other.x), y(other.y) { }
	Vec2t& operator=(const Vec2t &other) {
		if (this == &other) {
			return this;
		}

		x = other.x;
		y = other.y;
		return this;
	}
	Vec2t(Vec2t &&other) = delete;

	Vec2t operator+(const Vec2t &v) const {
		return Vec2t{ x + v.x, y + v.y };
	}

	Vec2t operator-(const Vec2t &v) const {
		return Vec2t{ x - v.x, y - v.y };
	}

	Vec2t operator*(T v) const {
		return Vec2t{ x * v, y * v };
	}

	Vec2t operator/(T v) const {
		return Vec2t{ x / v, y / v };
	}

	Vec2t operator-() const {
		return Vec2t{ -x, -y };
	}

	T cross(const Vec2t &v) const {
		return x * v.y - y * v.x;
	}

	T dot(const Vec2t &v) const {
		return x * v.x + y * v.y;
	}

	T lengthSqr() const {
		return dot(*this);
	}

	T length() const {
		return sqrt(lengthSqr());
	}

	Vec2t normalized() const {
		return *this / lenght();
	}
};

template <typename T>
struct Vec3t {
	union {
		struct {
			T x, y, z;
		};
		struct {
			T r, g, b;
		};
		struct {
			T u, v, w;
		};
		T data[3];
	};

	Vec3t() {
		memset(this, 0, 3 * sizeof(T));
	}

	Vec3t(T x) : x(x), y(x), z(x) { }

	Vec3t(T x, T y, T z) : x(x), y(y), z(z) { }

	Vec3t(const Vec3t &other) : x(other.x), y(other.y), z(other.z) { }
	Vec3t& operator=(const Vec3t &other) {
		if (this == &other) {
			return *this;
		}

		x = other.x;
		y = other.y;
		z = other.z;
		return *this;
	}
	Vec3t(Vec3t &&other) : x(other.x), y(other.y), z(other.z) { }

	Vec3t operator+(const Vec3t &v) const {
		return Vec3t{ x + v.x, y + v.y, z + v.z };
	}

	Vec3t operator-(const Vec3t &v) const {
		return Vec3t{ x - v.x, y - v.y, z - v.z };
	}

	Vec3t operator*(T v) const {
		return Vec3t{ x * v, y * v, z * v };
	}

	Vec3t operator/(T v) const {
		return Vec3t{ x / v, y / v, z / v };
	}

	Vec3t operator-() const {
		return Vec3t{ -x, -y, -z };
	}

	Vec3t cross(const Vec3t &v) const {
		return Vec3t{
			y * v.z - z * v.y,
			z * v.x - x * v.z,
			x * v.y - y * v.x
		};
	}

	T dot(const Vec3t &v) const {
		return x * v.x + y * v.y + z * v.z;
	}

	T lengthSqr() const {
		return dot(*this);
	}

	T length() const {
		return sqrt(lengthSqr());
	}

	Vec3t normalized() const {
		return *this / length();
	}
};

template <typename T>
struct Vec4t {
	union {
		struct {
			T x, y, z, w;
		};
		struct {
			T r, g, b, a;
		};
		struct {
			T u, v, w, t;
		};
		T data[4];
	};

	Vec4t() {
		memset(this, 0, 4 * sizeof(T));
	}

	Vec4t(T x) : x(x), y(x), z(x), w(x) { }

	Vec4t(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) { }

	Vec4t(const Vec3t<T> &v, T w = T(0)) : x(v.x), y(v.y), z(v.z), w(w) { }

	Vec4t(const Vec4t &other) : x(other.x), y(other.y), z(other.z), w(other.w) { }

	Vec4t& operator=(const Vec4t &other) {
		if (this == &other) {
			return *this;
		}

		x = other.x;
		y = other.y;
		z = other.z;
		w = other.w;
		return *this;
	}
	Vec4t(Vec4t &&other) : x(other.x), y(other.y), z(other.z), w(other.w) { }

	Vec4t operator+(const Vec4t &v) const {
		return Vec4t{ x + v.x, y + v.y, z + v.z, w + v.w };
	}

	Vec4t operator-(const Vec4t &v) const {
		return Vec4t{ x - v.x, y - v.y, z - v.z, w - v.w };
	}

	Vec4t operator*(T v) const {
		return Vec4t{ x * v, y * v, z * v, w * v };
	}

	Vec4t operator/(T v) const {
		return Vec4t{ x / v, y / v, z / v, w / v };
	}

	Vec4t operator-() const {
		return Vec4t{ -x, -y, -z, -w };
	}

	T dot(const Vec4t &v) const {
		return x * v.x + y * v.y + z * v.z + w * v.w;
	}

	T lengthSqr() const {
		return dot(*this);
	}

	T length() const {
		return sqrt(lengthSqr());
	}

	Vec4t normalized() const {
		return *this / lenght();
	}
};

template <class T>
struct Mat4t {
	using VecType = Vec4t<T>;

	union {
		struct {
			VecType row1, row2, row3, row4;
		};
		T data[16];
	};

	Mat4t() : Mat4t(T(1)) { }

	Mat4t(T unit) {
		T zero = T(0);
		row1 = VecType{ unit, zero, zero, zero };
		row2 = VecType{ zero, unit, zero, zero };
		row3 = VecType{ zero, zero, unit, zero };
		row4 = VecType{ zero, zero, zero, unit };
	}

	Mat4t(T data[16]) {
		copy(data);
	}

	Mat4t(const VecType columns[4]) {
		SizeType offset = 0;
		for (int i = 0; i < 4; ++i) {
			memcpy(this->data + offset, columns[i].data, sizeof(VecType));
			offset += 4;
		}
	}

	Mat4t(const VecType &row1, const VecType &row2, const VecType &row3, const VecType &row4) :
		row1(row1), row2(row2), row3(row3), row4(row4) {

	}

	Mat4t(const Mat4t &other) {
		copy(other.data);
	}

	Mat4t& operator=(const Mat4t &other) {
		if (this != &other) {
			copy(other.data);
		}

		return *this;
	}

	Mat4t transpose() const {
		return Mat4t{
			VecType { row1.x, row2.x, row3.x, row4.x },
			VecType { row1.y, row2.y, row3.y, row4.y },
			VecType { row1.z, row2.z, row3.z, row4.z },
			VecType { row1.w, row2.w, row3.w, row4.w },
		};
	}

	Mat4t translate(const Vec3t<T> &vec) {
		Mat4t result = *this;
		result.row1.w += vec.x;
		result.row2.w += vec.y;
		result.row3.w += vec.z;

		return result;
	}

	Mat4t rotate(const Vec3t<T> &vec, T deg) {
		T rad = radians(deg);
		T cos = std::cos(rad);
		T sin = std::sin(rad);

		Vec3t<T> axis = vec.normalized();
		T one = T(1);
		Vec3t<T> icos = (one - cos) * axis;
		Vec3t<T> isin = sin * axis;

		Mat4t result(one);

		result.data[0] = cos + icos.x * axis.x;
		result.data[1] = icos.x * axis.y - isin.z;
		result.data[2] = icos.x * axis.z + isin.y;

		result.data[4] = result.data[1] + 2 * isin.z;
		result.data[5] = cos + icos.y * axis.y;
		result.data[6] = icos.y * axis.z - isin.x;

		result.data[8] = result.data[2] - 2 * isin.y;
		result.data[9] = result.data[6] + 2 * isin.x;
		result.data[10] = cos + icos.z * axis.z;

		return result * *this;
	}

	Mat4t scale(const Vec3t<T> &scalar) {
		return Mat4t{
			row1 * scalar.x,
			row2 * scalar.y,
			row3 * scalar.z,
			row4
		};
	}

private:
	void copy(const T data[16]) {
		memcpy(this->data, data, sizeof(Mat4t));
	}
};

} // namespace /*Packed*/
#pragma pack(pop)

template <class VecT>
VecT normalized(const VecT vec) {
	return vec.normalized();
}

template <class T>
Mat4t<T> lookAt(const Vec3t<T> &target, const Vec3t<T> &pos, const Vec3t<T> &upTmp) {
	// LH view
	const Vec3t<T> in = (target - pos).normalized();
	const Vec3t<T> right = upTmp.cross(in);
	const Vec3t<T> up = in.cross(right);

	using VecType = Mat4t<T>::VecType;
	return Mat4t<T> {
		VecType(right, -right.dot(pos)),
			VecType(up, -up.dot(pos)),
			VecType(in, -in.dot(pos)),
			VecType(T(0), T(0), T(0), T(1))
	};
}

template <class T>
Mat4t<T> perspective(T FOV, T aspectRatio, T nearPlane, T farPlane) {
	const T theta = radians(FOV * T(0.5));
	const T cosTheta = std::cos(theta);
	const T sinTheta = std::sin(theta);
	const T one = T(1);

	const T height = cosTheta / sinTheta;
	const T width = height / aspectRatio;
	const T invDepth = farPlane / (farPlane - nearPlane);
	const T scaledInvDepth = -nearPlane * invDepth;

	Mat4t<T> result(one);
	result.row1.x = width;
	result.row2.y = height;
	result.row3.z = invDepth;
	result.row3.w = scaledInvDepth;
	result.row4.z = one;
	result.row4.w = T(0);

	return result;
}

// TODO
template <class T>
Mat4t<T> orthographic(T left, T right, T bottom, T top, T nearPlane, T farPlane) {
	auto xDivisor = 1 / (right - left);
	auto yDivisor = 1 / (top - bottom);
	auto zDivisor = 1 / (farPlane - nearPlane);

	Mat4t<T> result(T(1));
	result.row1.x = 2 * xDivisor;
	result.row1.w = -(right + left) * xDivisor;
	result.row2.y = 2 * yDivisor;
	result.row2.w = -(top + bottom) / yDivisor;
	result.row3.z = zDivisor;
	result.row3.w = -nearPlane * zDivisor;

	return result;
}

} // namespace dmath

template <class T>
dmath::Vec2t<T> operator*(T f, dmath::Vec2t<T> v) {
	return Vec2t{ v.x * f, v.y * f };
}

template <class T>
dmath::Vec3t<T> operator*(T f, dmath::Vec3t<T> v) {
	return dmath::Vec3t<T>{ v.x * f, v.y * f, v.z * f };
}

template <class T>
dmath::Mat4t<T> operator*(const dmath::Mat4t<T> &m1, const dmath::Mat4t<T> &m2) {
	using VecType = dmath::Mat4t<T>::VecType;

	dmath::Mat4t<T> m = m2.transpose();
	return dmath::Mat4t<T> {
		VecType{ m1.row1.dot(m.row1), m1.row1.dot(m.row2), m1.row1.dot(m.row3), m1.row1.dot(m.row4) },
			VecType{ m1.row2.dot(m.row1), m1.row2.dot(m.row2), m1.row2.dot(m.row3), m1.row2.dot(m.row4) },
			VecType{ m1.row3.dot(m.row1), m1.row3.dot(m.row2), m1.row3.dot(m.row3), m1.row3.dot(m.row4) },
			VecType{ m1.row4.dot(m.row1), m1.row4.dot(m.row2), m1.row4.dot(m.row3), m1.row4.dot(m.row4) },
	};
}

template <class T>
dmath::Vec4t<T> operator*(const dmath::Mat4t<T> &m, const dmath::Vec4t<T> &v) {
	return Vec3t<T> {
		m.row1.dot(v), m.row2.dot(v), m.row3.dot(v)
	};
}

template <class T>
dmath::Vec3t<T> operator*(const dmath::Mat4t<T> &m, const dmath::Vec3t<T> &vec) {
	auto v = Vec4t<T>(vec);
	v.w = T(1);
	return Vec3t<T> {
		m.row1.dot(v), m.row2.dot(v), m.row3.dot(v)
	};
}

using Vec2 = dmath::Vec2t<float>;
using Vec3 = dmath::Vec3t<float>;
using Vec4 = dmath::Vec4t<float>;

using Vec2i = dmath::Vec2t<int>;
using Vec3i = dmath::Vec3t<int>;
using Vec4i = dmath::Vec4t<int>;

using Mat = dmath::Mat4t<float>;
using Mat4 = Mat;

#ifdef __CUDACC__
}
#endif // __CUDACC__
