#pragma once

#include "cuda_cpu_common.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"

struct CudaRasterizer;

struct Drawable {
	virtual void draw(CudaRasterizer &renderer) const = 0;
};

struct Mesh : Drawable {
	Mesh(const char *filename, const char *shaderName);
	void draw(CudaRasterizer &renderer) const override;

	Vector<Vertex> geometry;
	mutable Vector<unsigned int> indices;
	Mat4 transform;
	String shader;
};

struct Scene : Drawable {
	Scene(const Vector<Mesh> &meshes) : meshes(meshes) { }

	void draw(CudaRasterizer &renderer) const override;

	Vector<Mesh> meshes;
};