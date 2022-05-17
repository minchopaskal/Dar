#pragma once

#include "utils/defines.h"
#include "math/dar_math.h"

#include "framework/camera.h"
#include "d3d12/command_list.h"
#include "d3d12/descriptor_heap.h"
#include "d3d12/resource_manager.h"

#include "gpu_cpu_common.hlsli"

using TextureId = unsigned int;
using MaterialId = SizeType;
using NodeId = SizeType;
using LightId = SizeType;
using CameraId = SizeType;

#define INVALID_TEXTURE_ID (unsigned int)(-1)
#define INVALID_MATERIAL_ID SizeType(-1)
#define INVALID_NODE_ID SizeType(-1)
#define INVALID_LIGHT_ID SizeType(-1)
#define INVALID_CAMERA_ID SizeType(-1)

enum class ConstantBufferView : unsigned int {
	MVPBuffer = 0,
	MeshData = 1,

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

struct Material {
	MaterialId id = INVALID_MATERIAL_ID;
	MaterialData materialData;
};

enum class TextureType : unsigned int {
	Invalid = 0,

	BaseColor,
	Normals,
	Metalness,
	Roughness,
	AmbientOcclusion,

	Count
};

enum class TextureFormat : unsigned int {
	Invalid = 0,

	RGBA_8BIT,

	Count
};

struct TextureDesc {
	String path;
	TextureId id = INVALID_TEXTURE_ID;
	TextureType type = TextureType::Invalid;
	TextureFormat format;
};

struct Mesh {
	Mat4 modelMatrix = Mat4(1.f);
	mutable Dar::ResourceHandle meshDataHandle = INVALID_RESOURCE_HANDLE;
	MaterialId mat = INVALID_MATERIAL_ID;
	SizeType indexOffset = 0;
	SizeType numIndices = 0;

	void uploadMeshData(Dar::UploadHandle uploadHandle) const;

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

	virtual void draw(Dar::CommandList &cmdList, const Scene &scene) const = 0;

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

	void draw(Dar::CommandList &cmdList, const Scene &scene) const override;

private:
	void updateMeshDataHandles() const;
};

struct LightNode : Node {
	LightData lightData;

	LightNode() {
		lightData.type = LightType::Invalid;
		nodeType = NodeType::Light;
	}

	void draw(Dar::CommandList &, const Scene &scene) const override final {
		// TODO: debug draw point lights
		// IDEA: debug draw dir lights by giving them position
	}
};

// TODO: camera node should be able to be attached to a parent node, controlling it's position
struct CameraNode : Node {
	CameraNode(Dar::Camera &&camera) {
		nodeType = NodeType::Camera;
		this->camera = std::move(camera);
	}

	virtual void draw(Dar::CommandList &, const Scene &) const override final {}

	Dar::Camera *getCamera() {
		return &camera;
	}

private:
	Dar::Camera camera;
};

struct Vertex {
	Vec3 pos;
	Vec3 normal;
	Vec3 tangent;
	Vec2 uv;
};

// TODO: encapsulate members
struct Scene {
	Vector<Node *> nodes; ///< Vector with pointers to all nodes in the scene
	Vector<LightId> lightIndices; ///< Indices of the lights in the nodes vector
	Vector<CameraId> cameraIndices; ///< Indices of the cameras in the nodes vector
	Vector<Material> materials; ///< Vector with all materials in the scene
	Vector<TextureDesc> textures; ///< Vector with all textures in the scene. Meshes could share texture ids.
	Vector<Vertex> vertices; ///< All vertices in the scene.
	Vector<unsigned int> indices; ///< All indices for all meshes, indexing in the vertices array.
	Vector<Dar::ResourceHandle> textureHandles;
	Dar::ResourceHandle materialsHandle; ///< Handle to the GPU buffer holding all materials' data.
	Dar::ResourceHandle lightsHandle; ///< Handle to the GPU buffer holding all lights' data.
	BBox sceneBox;
	unsigned int renderCamera = 0; ///< Id of the camera used for rendering

	Scene();
	~Scene();

	bool setCameraForCameraController(Dar::ICameraController &controller) {
		static const int activeCameraIdx = 0; // TODO: this should not be hardcoded

		if (cameraIndices.empty()) {
			return false;
		}

		CameraNode *camNode = dynamic_cast<CameraNode *>(nodes[cameraIndices[activeCameraIdx]]);
		if (camNode == nullptr) {
			return false;
		}

		controller.setCamera(camNode->getCamera());
		return true;
	}

	SizeType getNumLights() const {
		return lightIndices.size();
	}

	LightId addNewLight(LightNode *l) {
		LightId id = nodes.size();

		l->id = id;
		nodes.push_back(l);
		lightIndices.push_back(id);

		lightsNeedUpdate = changesSinceLastCheck = true;

		return id;
	}

	CameraId addNewCamera(CameraNode *cam) {
		CameraId id = nodes.size();

		cam->id = id;
		nodes.push_back(cam);
		cameraIndices.push_back(id);

		return id;
	}

	MaterialId getNewMaterial(MaterialData materialData) {
		Material m;
		m.id = materials.size();
		m.materialData = materialData;
		materials.push_back(m);

		materialsNeedUpdate = changesSinceLastCheck = true;

		return m.id;
	}

	TextureId getNewTexture(const char *path, TextureType type) {
		for (int i = 0; i < textures.size(); ++i) {
			if (strcmp(textures[i].path.c_str(), path) == 0) {
				return textures[i].id;
			}
		}

		TextureDesc res = { String{path}, static_cast<unsigned int>(textures.size()), type };
		textures.push_back(res);
		textureHandles.push_back({});

		texturesNeedUpdate = changesSinceLastCheck = true;

		return res.id;
	}

	const Material &getMaterial(MaterialId id) const {
		dassert(id >= 0 && id < materials.size() && id != INVALID_MATERIAL_ID);
		return materials[id];
	}

	const TextureDesc &getTexture(TextureId id) const {
		dassert(id >= 0 && id < textures.size() && id != INVALID_TEXTURE_ID);
		return textures[id];
	}

	const void *getVertexBuffer() const {
		return &vertices[0];
	}

	const void *getIndexBuffer() const {
		return &indices[0];
	}

	const UINT getVertexBufferSize() const {
		SizeType sz = vertices.size() == 0 ? 0 : vertices.size() * sizeof(vertices[0]);
		dassert(sz < ((SizeType(1) << 32) - 1));
		return static_cast<UINT>(sz);
	}

	const UINT getIndexBufferSize() const {
		SizeType sz = indices.size() == 0 ? 0 : indices.size() * sizeof(indices[0]);
		dassert(sz < ((SizeType(1) << 32) - 1));
		return static_cast<UINT>(sz);
	}

	Dar::Camera *getRenderCamera() {
		return dynamic_cast<CameraNode *>(nodes[cameraIndices[renderCamera]])->getCamera();
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

	bool uploadSceneData(Dar::UploadHandle uploadHandle);

	void draw(Dar::CommandList &cmdList) const;

	bool hadChangesSinceLastCheck() const {
		bool result = changesSinceLastCheck;
		changesSinceLastCheck = false;

		return result;
	}

private:
	bool uploadLightData(Dar::UploadHandle uploadHandle);
	bool uploadMaterialData(Dar::UploadHandle uploadHandle);
	bool uploadTextureData(Dar::UploadHandle uploadHandle);

	void drawNodeImpl(Node *node, Dar::CommandList &cmdList, const Scene &scene, DynamicBitset &drawnNodes) const;

private:
	Dar::HeapHandle texturesHeap; ///< Heap of the memory holding the textures' data

	bool texturesNeedUpdate; ///< Indicates textures have been changed and need to be reuploaded to the GPU.
	bool lightsNeedUpdate; ///< Indicates lights have been changed and need to be reuploaded to the GPU.
	bool materialsNeedUpdate; ///< Indicates materials have been changed and need to be reuploaded to the GPU.
	mutable bool changesSinceLastCheck; ///< Check if any changes in the textures/lights/materials was done since the last read of this value.
};
