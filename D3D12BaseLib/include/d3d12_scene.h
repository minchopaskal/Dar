#pragma once

#include "d3d12_defines.h"
#include "d3d12_math.h"

#include "d3d12_camera.h"
#include "d3d12_command_list.h"
#include "d3d12_resource_manager.h"

using TextureId = SizeType;
using MaterialId = SizeType;
using NodeId = SizeType;
using LightId = SizeType;
using CameraId = SizeType;

#define INVALID_MATERIAL_ID SizeType(-1)
#define INVALID_TEXTURE_ID SizeType(-1)
#define INVALID_NODE_ID SizeType(-1)
#define INVALID_LIGHT_ID SizeType(-1)
#define INVALID_CAMERA_ID SizeType(-1)

enum class ConstantBufferView : unsigned int {
	MVPBuffer = 0,
	LightsBuffer = 1,
	MeshData = 2,

	Count
};

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
struct GPUMaterial {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int normals;
};

struct Material {
	MaterialId id = INVALID_MATERIAL_ID;
	TextureId diffuse = INVALID_TEXTURE_ID;
	TextureId specular = INVALID_TEXTURE_ID;
	TextureId normals = INVALID_TEXTURE_ID;
};

enum class TextureType : unsigned int {
	Invalid = 0,

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
	Mat4 modelMatrix = Mat4(1.f);
	mutable ResourceHandle meshDataHandle = INVALID_RESOURCE_HANDLE;
	MaterialId mat = INVALID_MATERIAL_ID;
	SizeType indexOffset;
	SizeType numIndices;

	void uploadMeshData(UploadHandle uploadHandle) const;

private:
	mutable Mat4 cache = Mat4(1.f);
};

struct Scene;

enum class NodeType : int {
	Invalid = -1,

	Camera = 0,
	Light,
	Model,

	Count
};

struct Node {
	Vector<NodeId> children;
	NodeId id = INVALID_NODE_ID;

	virtual ~Node() {}

	virtual void draw(CommandList &cmdList, const Scene &scene) const = 0;

	NodeType getNodeType() const {
		return nodeType;
	}

protected:
	NodeType nodeType = NodeType::Invalid;
};

struct ModelNode : Node {
	Vector<Mesh> meshes;

	ModelNode() {
		nodeType = NodeType::Model;
	}

	void draw(CommandList &cmdList, const Scene &scene) const override;

private:
	void updateMeshDataHandles() const;
};

enum class LightType : int {
	Invalid = -1,

	Point = 0,
	Directional,
	Spot,

	Count
};

struct GPULight {
	Vec3 position;
	Vec3 diffuse;
	Vec3 ambient;
	Vec3 specular;
	Vec3 attenuation;
	Vec3 direction;
	float innerAngleCutoff;
	float outerAngleCutoff;
	int type;
};

struct LightNode : Node {
	Vec3 position;
	Vec3 diffuse;
	Vec3 ambient;
	Vec3 specular;
	Vec3 attenuation;
	Vec3 direction;
	float innerAngleCutoff;
	float outerAngleCutoff;
	LightType type;

	LightNode() : innerAngleCutoff(0.f), outerAngleCutoff(0.f), type(LightType::Invalid) {
		nodeType = NodeType::Light;
	}

	void draw(CommandList&, const Scene &scene) const override final {
		// TODO: debug draw point lights
		// IDEA: debug draw dir lights by giving them position
	}
};

struct CameraNode : Node {
	CameraNode(Camera &&camera) {
		nodeType = NodeType::Camera;
		camera = std::move(camera); // TODO: check move constructor
	}
	
	virtual void draw(CommandList &, const Scene&) const override final { }

private:
	Camera camera;
};

struct Vertex {
	Vec3 pos;
	Vec3 normal;
	Vec2 uv;
};

// TODO: encapsulate members
struct Scene {
	Vector<Node*> nodes; ///< Vector with pointers to all nodes in the scene
	Vector<LightId> lightIndices; ///< Indices of the lights in the nodes vector
	Vector<CameraId> cameraIndices; ///< Indices of the cameras in the nodes vector
	Vector<Material> materials; ///< Vector with all materials in the scene
	Vector<Texture> textures; ///< Vector with all textures in the scene. Meshes could share texture ids.
	Vector<Vertex> vertices; ///< All vertices in the scene.
	Vector<unsigned int> indices; ///< All indices for all meshes, indexing in the vertices array.
	ResourceHandle materialsHandle; ///< Handle to the GPU buffer holding all materials' data.
	ResourceHandle lightsHandle; ///< Handle to the GPU buffer holding all lights' data.
	BBox sceneBox;
	CameraId renderCamera = 0; ///< Id of the camera used for rendering

	Scene();

	SizeType getNumLights() const {
		return lightIndices.size();
	}

	LightId addNewLight(LightNode *l) {
		LightId id = nodes.size();
		
		l->id = id;
		nodes.push_back(l);
		lightIndices.push_back(id);

		lightsNeedUpdate = true;

		return id;
	}

	CameraId addNewCamera(CameraNode *cam) {
		CameraId id = nodes.size();

		cam->id = id;
		nodes.push_back(cam);
		lightIndices.push_back(id);

		return id;
	}

	MaterialId getNewMaterial(TextureId diffuse, TextureId specular, TextureId normals) {
		Material m;
		m.id = materials.size();
		m.diffuse = diffuse;
		m.specular = specular;
		m.normals = normals;
		materials.push_back(m);

		materialsNeedUpdate = true;

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

	const Material& getMaterial(MaterialId id) const {
		dassert(id >= 0 && id < materials.size() && id != INVALID_MATERIAL_ID);
		return materials[id];
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

	const SizeType getNumNodes() const {
		return nodes.size();
	}

	const SizeType getNumTextures() const {
		return textures.size();
	}

	const SizeType getNumMaterials() const {
		return materials.size();
	}

	// TODO: this is shit. Find a better way.
	void uploadSceneData();

	void draw(CommandList &cmdList) const;

private:
	void uploadLightData(UploadHandle uploadHandle);
	void uploadMaterialData(UploadHandle uploadHandle);

	void drawNodeImpl(Node *node, CommandList &cmdList, const Scene &scene, DynamicBitset &drawnNodes) const;

private:
	bool lightsNeedUpdate; ///< Indicates lights have been changed and need to be reuploaded to the GPU.
	bool materialsNeedUpdate; ///< Indicates materials have been changed and need to be reuploaded to the GPU.
};
