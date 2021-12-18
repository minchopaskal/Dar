#pragma once

#include "d3d12_defines.h"
#include "d3d12_math.h"

#include "d3d12_command_list.h"

using TextureId = SizeType;
using MaterialId = SizeType;
using NodeId = SizeType;

#define INVALID_MATERIAL_ID SizeType(-1)
#define INVALID_TEXTURE_ID SizeType(-1)
#define INVALID_NODE_ID SizeType(-1)

struct BBox {
	Vec3 pmin = BBox::invalidMinPoint();
	Vec3 pmax = BBox::invalidMaxPoint();

	static Vec3 invalidMinPoint() {
		return Vec3{ 1e20f, 1e20f, 1e20f };
	}

	static Vec3 invalidMaxPoint() {
		return Vec3{ -1e20f, -1e20f, -1e20f };
	}

	static BBox invalidBBox() {
		return BBox{ invalidMinPoint(), invalidMaxPoint() };
	}
};

// TODO: make pbr ofc
struct Material {
	MaterialId id = INVALID_MATERIAL_ID;
	TextureId diffuse = INVALID_TEXTURE_ID;
	TextureId specular = INVALID_TEXTURE_ID;
	TextureId normals = INVALID_TEXTURE_ID;
};

enum class TextureType : int {
	Invalid,

	Diffuse,
	Specular,
	Normals,

	Count
};

struct Texture {
	String path;
	TextureId id = INVALID_TEXTURE_ID;
	TextureType type = TextureType::Invalid;
};

struct Mesh {
	MaterialId mat = INVALID_MATERIAL_ID;
	SizeType indexOffset;
	SizeType numIndices;
};

struct Scene;

struct Node {
	Vector<NodeId> children;
	NodeId id = INVALID_NODE_ID;

	virtual void draw(CommandList &cmdList, const Scene &scene) const = 0;
};

struct Model : Node {
	Vector<Mesh> meshes;

	void draw(CommandList &cmdList, const Scene &scene) const override;
};

// TODO: implement cameras import
struct Camera : Node {
	void draw(CommandList&, const Scene &scene) const override { }
};

// TODO: implement lights import
struct Light : Node {
	void draw(CommandList&, const Scene &scene) const override { }
};

struct Vertex {
	Vec3 pos;
	Vec3 normal;
	Vec2 uv;
};

// TODO: encapsulate members
struct Scene {
	Vector<Node*> nodes;
	Vector<Material> materials;
	Vector<Texture> textures;
	Vector<Vertex> vertices;
	Vector<unsigned int> indices;
	BBox sceneBox;

	MaterialId getNewMaterial() {
		Material m;
		m.id = materials.size();
		materials.push_back(m);
		return m.id;
	}

	TextureId getNewTexture(const char *path, TextureType type) {
		for (int i = 0; i < textures.size(); ++i) {
			if (strcmp(textures[i].path.c_str(), path) == 0) {
				return textures[i].id;
			}
		}

		Texture res = { String{path}, textures.size(), type };
		textures.push_back(res);
		return res.id;
	}

	Material& getMaterial(MaterialId id) {
		dassert(id >= 0 && id < materials.size() && id != INVALID_MATERIAL_ID);
		return materials[id];
	}

	const Material& getMaterial(MaterialId id) const {
		dassert(id >= 0 && id < materials.size() && id != INVALID_MATERIAL_ID);
		return materials[id];
	}

	Texture& getTexture(TextureId id) {
		dassert(id >= 0 && id < textures.size() && id != INVALID_TEXTURE_ID);
		return textures[id];
	}

	const Texture& getTexture(TextureId id) const {
		dassert(id >= 0 && id < textures.size() && id != INVALID_TEXTURE_ID);
		return textures[id];
	}

	const void* getVertexBuffer() const {
		return &vertices[0];
	}

	const void* getIndexBuffer() const {
		return &indices[0];
	}

	const SizeType getVertexBufferSize() const {
		return vertices.size() == 0 ? 0 : vertices.size() * sizeof(vertices[0]);
	}

	const SizeType getIndexBufferSize() const {
		return indices.size() == 0 ? 0 : indices.size() * sizeof(indices[0]);
	}

	const SizeType getNumTextures() const {
		return textures.size();
	}

	void draw(CommandList &cmdList) const;

private:
	void drawNodeImpl(Node *node, CommandList &cmdList, const Scene &scene) const;
};
