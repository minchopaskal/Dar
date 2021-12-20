#pragma once

#include <cmath>
#include <type_traits>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

template <class T>
using ConstRefT = typename std::conditional<
	sizeof(T) <= sizeof(void*), 
	typename std::remove_reference_t<typename std::remove_const_t<T>>,
	typename 
		std::add_const_t<
			std::remove_reference_t<
				std::remove_const_t<T>
			>
		>&
>::type;

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

template <class T>
bool areEqual(ConstRefT<T> a, ConstRefT<T> b) {
	if constexpr (std::is_floating_point<T>::value) {
		return std::abs(a - b) < T(1e-6);
	} else {
		return a == b;
	}
}

#pragma pack(push, 1)
namespace Packed {

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
			return *this;
		}

		x = other.x;
		y = other.y;
		return *this;
	}
	Vec2t(Vec2t &&other) : x(other.x), y(other.y) { }

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

	[[nodiscard]] Vec3t operator+(const Vec3t &v) const {
		return Vec3t{ x + v.x, y + v.y, z + v.z };
	}

	[[nodiscard]] Vec3t operator-(const Vec3t &v) const {
		return Vec3t{ x - v.x, y - v.y, z - v.z };
	}

	[[nodiscard]] Vec3t operator*(T v) const {
		return Vec3t{ x * v, y * v, z * v };
	}

	[[nodiscard]] Vec3t operator/(T v) const {
		return Vec3t{ x / v, y / v, z / v };
	}

	[[nodiscard]] Vec3t operator-() const {
		return Vec3t{ -x, -y, -z };
	}

	[[nodiscard]] Vec3t cross(const Vec3t &v) const {
		return Vec3t{
			y * v.z - z * v.y,
			z * v.x - x * v.z,
			x * v.y - y * v.x
		};
	}

	[[nodiscard]] T dot(const Vec3t &v) const {
		return x * v.x + y * v.y + z * v.z;
	}

	[[nodiscard]] T lengthSqr() const {
		return dot(*this);
	}

	[[nodiscard]] T length() const {
		return sqrt(lengthSqr());
	}

	[[nodiscard]] Vec3t normalized() const {
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

	[[nodiscard]] Vec4t operator+(const Vec4t &v) const {
		return Vec4t{ x + v.x, y + v.y, z + v.z, w + v.w };
	}

	[[nodiscard]] Vec4t operator-(const Vec4t &v) const {
		return Vec4t{ x - v.x, y - v.y, z - v.z, w - v.w };
	}

	[[nodiscard]] Vec4t operator*(T v) const {
		return Vec4t{ x * v, y * v, z * v, w * v };
	}

	[[nodiscard]] Vec4t operator/(T v) const {
		return Vec4t{ x / v, y / v, z / v, w / v };
	}

	[[nodiscard]] Vec4t operator-() const {
		return Vec4t{ -x, -y, -z, -w };
	}

	[[nodiscard]] T dot(const Vec4t &v) const {
		return x * v.x + y * v.y + z * v.z + w * v.w;
	}

	[[nodiscard]] T lengthSqr() const {
		return dot(*this);
	}

	[[nodiscard]] T length() const {
		return sqrt(lengthSqr());
	}

	[[nodiscard]] Vec4t normalized() const {
		return *this / lenght();
	}
};

template <class T>
struct Mat3t {
	using VecType = Vec3t<T>;

	union {
		struct {
			VecType row1, row2, row3;
		};
		T data[9];
	};

	Mat3t() : Mat3t(T(1)) { }

	Mat3t(T unit) {
		T zero = T(0);
		row1 = VecType{ unit, zero, zero };
		row2 = VecType{ zero, unit, zero };
		row3 = VecType{ zero, zero, unit };
	}

	Mat3t(T data[9]) {
		copy(data);
	}

	Mat3t(const VecType rows[3]) {
		SizeType offset = 0;
		for (int i = 0; i < 3; ++i) {
			memcpy(this->data + offset, rows[i].data, sizeof(VecType));
			offset += 3;
		}
	}

	Mat3t(const VecType &row1, const VecType &row2, const VecType &row3) :
		row1(row1), row2(row2), row3(row3) {

	}

	Mat3t(const Mat3t &other) {
		copy(other.data);
	}

	Mat3t& operator=(const Mat3t &other) {
		if (this != &other) {
			copy(other.data);
		}

		return *this;
	}

	[[nodiscard]] Mat3t inverse() const {
		// TODO:
		return Mat3t(T(1));
	}

	[[nodiscard]] Mat3t transpose() const {
		return Mat3t{
			VecType { row1.x, row2.x, row3.x },
			VecType { row1.y, row2.y, row3.y },
			VecType { row1.z, row2.z, row3.z },
		};
	}

	[[nodiscard]] Mat3t rotate(const Vec3t<T> &vec, T deg) {
		// TODO:
		return Mat3t(T(1));
	}

	[[nodiscard]] Mat3t scale(const Vec3t<T> &scalar) {
		return Mat3t{
			row1 * scalar.x,
			row2 * scalar.y,
			row3 * scalar.z
		};
	}

private:
	void copy(const T data[9]) {
		memcpy(this->data, data, sizeof(Mat3t));
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

	Mat4t(const Mat3t<T> &rotationMatrix, const VecType &translationVector) :
		row1(rotationMatrix.row1), row2(rotationMatrix.row2), row3(rotationMatrix.row3), row4(translationVector) {

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

	[[nodiscard]] Mat4t inverse() const {
		// TODO:
		return Mat4t(T(1));
	}

	[[nodiscard]] Mat4t transpose() const {
		return Mat4t{
			VecType { row1.x, row2.x, row3.x, row4.x },
			VecType { row1.y, row2.y, row3.y, row4.y },
			VecType { row1.z, row2.z, row3.z, row4.z },
			VecType { row1.w, row2.w, row3.w, row4.w },
		};
	}

	[[nodiscard]] Mat4t translate(const Vec3t<T> &vec) {
		Mat4t result = *this;
		result.row1.w += vec.x;
		result.row2.w += vec.y;
		result.row3.w += vec.z;

		return result;
	}

	[[nodiscard]] Mat4t rotate(const Vec3t<T> &vec, T deg) {
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

	[[nodiscard]] Mat4t scale(const Vec3t<T> &scalar) {
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

template <class T>
struct Quatt {
	/// Create a quaternion from an angle in degrees and a rotation axis
	static Quatt makeQuat(T angle, const Vec3t<T> &axis) {
		dassert(areEqual<T>(axis.lengthSqr(), T(1))); // make sure the axis is normalized

		angle = radians(angle);

		const T sinTheta = sin(angle / 2);
		const T scalarPart = cos(angle / 2);
		const Vec3t<T> vecPart = axis * sinTheta;
		return Quatt(vecPart, scalarPart);
	}

	Quatt(Quatt&) = default;
	Quatt& operator=(Quatt&) = default;

	[[nodiscard]] Mat3t<T> getRotationMatrix() const {
		const T one = T(1);
		const T two = T(2);
		return Mat3t<T> {
			Vec3t<T>{
				one - two * v.y * v.y - two * v.z * v. z,
				two * v.x * v.y - two * s * v.z,
				two * v.x * v.z + two * s * v.y
			},
			Vec3t<T>{
				two * v.x * v.y + two * s * v.z,
				one - two * v.x * v.x - two * v.z * v.z,
				two * v.y * v.z - two * s * v.x
			},
			Vec3t<T>{
				two * v.x * v.z - two * s * v.y,
				two * v.y * v.z + two * s * v.x,
				one - two * v.x * v.x - two * v.y * v.y
			}
		};
	}
	
	[[nodiscard]] Quatt operator*(const Quatt &q) const {
		return Quatt(
			Vec3t<T>{ q.s * v + s * q.v + v.cross(q.v) },
			s * q.s - v.dot(q.v)
		);
	}

	[[nodiscard]] Quatt conjugate() const {
		return Quatt{ -v.x, -v.y, -v.z, s };
	}

	[[nodiscard]] Vec3t<T> rotate(const Vec3t<T> &v) {
		Quatt q1 = *this * Quatt(v, T(0));
		Quatt q2 = q1 * conjugate();
		return q2.v;
	}

private:
	Quatt() : v(T(0)), s(T(0)) { }
	Quatt(T x, T y, T z, T w) : v(x, y, z), s(w) { }
	Quatt(const Vec3t<T> &v, T s) : v(v), s(s) { }
	
private:
	Vec3t<T> v;
	T s;
};

} // namespace /*Packed*/
#pragma pack(pop)

template <class VecT>
[[nodiscard]] VecT normalized(const VecT vec) {
	return vec.normalized();
}

template <class T>
Packed::Mat4t<T> lookAt(const Packed::Vec3t<T> &target, const Packed::Vec3t<T> &pos, const Packed::Vec3t<T> &upTmp) {
	// LH view
	const Packed::Vec3t<T> in = (target - pos).normalized();
	const Packed::Vec3t<T> right = upTmp.cross(in);
	const Packed::Vec3t<T> up = in.cross(right);

	using VecType = Packed::Mat4t<T>::VecType;
	return Packed::Mat4t<T> {
		VecType(right, -right.dot(pos)),
		VecType(up, -up.dot(pos)),
		VecType(in, -in.dot(pos)),
		VecType(T(0), T(0), T(0), T(1))
	};
}

template <class T>
Packed::Mat4t<T> perspective(T FOV, T aspectRatio, T nearPlane, T farPlane) {
	const T theta = radians(FOV * T(0.5));
	const T cosTheta = std::cos(theta);
	const T sinTheta = std::sin(theta);
	const T one = T(1);

	const T height = cosTheta / sinTheta;
	const T width = height / aspectRatio;
	const T invDepth = farPlane / (farPlane - nearPlane);
	const T scaledInvDepth = -nearPlane * invDepth;

	Packed::Mat4t<T> result(one);
	result.row1.x = width;
	result.row2.y = height;
	result.row3.z = invDepth;
	result.row3.w = scaledInvDepth;
	result.row4.z = one;
	result.row4.w = T(0);

	return result;
}

template <class T>
Packed::Mat4t<T> orthographic(T left, T right, T bottom, T top, T nearPlane, T farPlane) {
	auto xDivisor = 1 / (right - left);
	auto yDivisor = 1 / (top - bottom);
	auto zDivisor = 1 / (farPlane - nearPlane);

	Packed::Mat4t<T> result(T(1));
	result.row1.x = 2 * xDivisor;
	result.row1.w = -(right + left) * xDivisor;
	result.row2.y = 2 * yDivisor;
	result.row2.w = -(top + bottom) / yDivisor;
	result.row3.z = zDivisor;
	result.row3.w = -nearPlane * zDivisor;

	return result;
}

template <template <class> class VecType, class T>
VecType<T> lerp(const VecType<T> &v1, const VecType<T> &q2, T t) {

}

template <class T>
Packed::Quatt<T> slerp(const Packed::Quatt<T> &q1, const Packed::Quatt<T> &q2, T t) {

}

} // namespace dmath

template <class T>
dmath::Packed::Vec2t<T> operator*(T f, dmath::Packed::Vec2t<T> v) {
	return dmath::Packed::Vec2t{ v.x * f, v.y * f };
}

template <class T>
dmath::Packed::Vec3t<T> operator*(T f, dmath::Packed::Vec3t<T> v) {
	return dmath::Packed::Vec3t<T>{ v.x * f, v.y * f, v.z * f };
}

template <class T>
dmath::Packed::Mat4t<T> operator*(const dmath::Packed::Mat4t<T> &m1, const dmath::Packed::Mat4t<T> &m2) {
	using VecType = dmath::Packed::Mat4t<T>::VecType;

	dmath::Packed::Mat4t<T> m = m2.transpose();
	return dmath::Packed::Mat4t<T> {
		VecType{ m1.row1.dot(m.row1), m1.row1.dot(m.row2), m1.row1.dot(m.row3), m1.row1.dot(m.row4) },
		VecType{ m1.row2.dot(m.row1), m1.row2.dot(m.row2), m1.row2.dot(m.row3), m1.row2.dot(m.row4) },
		VecType{ m1.row3.dot(m.row1), m1.row3.dot(m.row2), m1.row3.dot(m.row3), m1.row3.dot(m.row4) },
		VecType{ m1.row4.dot(m.row1), m1.row4.dot(m.row2), m1.row4.dot(m.row3), m1.row4.dot(m.row4) },
	};
}

template <class T>
dmath::Packed::Vec4t<T> operator*(const dmath::Packed::Mat4t<T> &m, const dmath::Packed::Vec4t<T> &v) {
	return dmath::Packed::Vec4t<T> {
		m.row1.dot(v), m.row2.dot(v), m.row3.dot(v), m.row4.dot(v)
	};
}

template <class T>
dmath::Packed::Vec3t<T> operator*(const dmath::Packed::Mat4t<T> &m, const dmath::Packed::Vec3t<T> &vec) {
	auto v = dmath::Packed::Vec4t<T>(vec);
	v.w = T(1);
	return dmath::Packed::Vec3t<T> {
		m.row1.dot(v), m.row2.dot(v), m.row3.dot(v)
	};
}

template <class T>
bool operator==(const dmath::Packed::Vec4t<T> &v1, const dmath::Packed::Vec4t<T> &v2) {
	return dmath::areEqual<T>(v1.x, v2.x) && dmath::areEqual<T>(v1.y, v2.y) && dmath::areEqual<T>(v1.z, v2.z) && dmath::areEqual<T>(v1.w, v2.w);
}

template <class T>
bool operator==(const dmath::Packed::Mat4t<T> &m1, const dmath::Packed::Mat4t<T> &m2) {
	return m1.row1 == m2.row1 && m1.row2 == m2.row2 && m1.row3 == m2.row3 && m1.row4 == m2.row4;
}

using Vec2 = dmath::Packed::Vec2t<float>;
using Vec3 = dmath::Packed::Vec3t<float>;
using Vec4 = dmath::Packed::Vec4t<float>;

using Vec2i = dmath::Packed::Vec2t<int>;
using Vec3i = dmath::Packed::Vec3t<int>;
using Vec4i = dmath::Packed::Vec4t<int>;

using Quat = dmath::Packed::Quatt<float>;

using Mat = dmath::Packed::Mat4t<float>;
using Mat3 = dmath::Packed::Mat3t<float>;
using Mat4 = Mat;
