#pragma once

#include "texture_utils.h"

#include "d3d12/command_list.h"
#include "d3d12/descriptor_heap.h"
#include "d3d12/resource_manager.h"
#include "framework/camera.h"
#include "graphics/renderer.h"
#include "math/dar_math.h"

#include "gpu_cpu_common.hlsli"

using MaterialId = SizeType;
using MeshId = SizeType;
using NodeId = SizeType;
using LightId = SizeType;
using CameraId = SizeType;

#define INVALID_MATERIAL_ID SizeType(-1)
#define INVALID_NODE_ID SizeType(-1)
#define INVALID_LIGHT_ID SizeType(-1)
#define INVALID_CAMERA_ID SizeType(-1)

enum class DefaultConstantBufferView : unsigned int {
	SceneData = 0,
	MeshData = 1,

	Count
};

enum class ShadowMapConstantBufferView : unsigned int {
	SceneData = 0,
	MeshData = 1,
	LightcasterDesc = 2,

	Count
};

struct BoundingSphere {
	Vec3 center;
	float radius;
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

	void addPoint(const Vec3 &p) {
		for (int i = 0; i < 3; ++i) {
			pmin[i] = glm::min(pmin[i], p[i]);
			pmax[i] = glm::max(pmax[i], p[i]);
		}
	}

	BoundingSphere getBoundingSphere() {
		Vec3 diameter = pmax - pmin;

		BoundingSphere result;
		result.center = pmin + Vec3(0.5f) * diameter;
		result.radius = glm::length(diameter) * 0.5f;

		return result;
	}
};

struct Material {
	MaterialId id = INVALID_MATERIAL_ID;
	MaterialData materialData;
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

	virtual void draw(Dar::FrameData &frameData, const Scene &scene) const = 0;

	NodeType getNodeType() const {
		return nodeType;
	}

protected:
	NodeType nodeType = NodeType::Invalid;
};

struct ModelNode : Node {
	MeshId startMesh = MeshId(-1);
	SizeType numMeshes = 0;

	ModelNode() {
		nodeType = NodeType::Model;
	}

	void draw(Dar::FrameData &frameData, const Scene &scene) const override;

private:
	void updateMeshDataHandles(const Scene &scene) const;
};

struct LightNode : Node {
	LightData lightData;

	LightNode() {
		lightData.type = LightType::InvalidLight;
		nodeType = NodeType::Light;
	}

	static LightNode directional(
		const Vec3 &direction,
		const Vec3 &diffuse,
		const Vec3 &ambient,
		const Vec3 &specular
	) {
		LightNode res;
		auto &l = res.lightData;

		l.type = LightType::Directional;
		l.direction = glm::normalize(direction);
		l.diffuse = diffuse;
		l.ambient = ambient;
		l.specular = specular;
		
		return res;
	}

	static LightNode spot(
		const Vec3 &diffuse,
		const Vec3 &ambient,
		const Vec3 &specular,
		float innerAngleCutoffRadians,
		float outerAngleCutoffRadians
	) {
		LightNode res;
		auto &l = res.lightData;

		l.type = LightType::Spot;
		l.diffuse = diffuse;
		l.ambient = ambient;
		l.specular = specular;
		l.innerAngleCutoff = cos(innerAngleCutoffRadians);
		l.outerAngleCutoff = cos(outerAngleCutoffRadians);

		return res;
	}

	static LightNode point(
		const Vec3 &position,
		const Vec3 &diffuse,
		const Vec3 &ambient,
		const Vec3 &specular,
		const Vec3 &attenuation
	) {
		LightNode res;
		auto &l = res.lightData;

		l.type = LightType::Point;
		l.position = position;
		l.diffuse = diffuse;
		l.ambient = ambient;
		l.specular = specular;
		l.attenuation = attenuation;

		return res;
	}

	void draw(Dar::FrameData&, const Scene&) const override final {
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

	virtual void draw(Dar::FrameData &, const Scene &) const override final {}

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

// Scene structure. Not very cache friendly, especially with these nodes on the heap :/
struct Scene {
	Vector<Node*> nodes; ///< Vector with pointers to all nodes in the scene
	Vector<LightId> lightIndices; ///< Indices of the lights in the nodes vector
	Vector<CameraId> cameraIndices; ///< Indices of the cameras in the nodes vector
	Vector<Mesh> meshes; ///< Vector will all the meshes in the scene.
	Vector<Material> materials; ///< Vector with all materials in the scene
	Vector<TextureDesc> textureDescs; ///< Vector with all textures in the scene. Meshes could share texture ids.
	Vector<Vertex> vertices; ///< All vertices in the scene.
	Vector<unsigned int> indices; ///< All indices for all meshes, indexing in the vertices array.
	Vector<Dar::TextureResource> textures;
	Dar::DataBufferResource materialsBuffer; ///< GPU buffer holding all materials' data.
	Dar::DataBufferResource lightsBuffer; ///< GPU buffer holding all lights' data.
	Dar::DataBufferResource lightcasterDescs[MAX_SHADOW_MAPS_COUNT]; ///< Indices of the lights casting shadows
	BBox sceneBox;
	int lightcasterIndices[MAX_SHADOW_MAPS_COUNT];
	unsigned int renderCamera = 0; ///< Id of the camera used for rendering

	Scene();
	~Scene();

	bool setCameraForCameraController(Dar::ICameraController &controller) {
		const int activeCameraIdx = 0; // TODO: this should not be hardcoded

		if (cameraIndices.empty()) {
			return false;
		}

		const auto camIdx = cameraIndices[activeCameraIdx];
		Node *baseCamNode = nodes[camIdx];

		CameraNode *camNode = dynamic_cast<CameraNode *>(baseCamNode);
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

	TextureId getNewTexture(const char *path) {
		for (int i = 0; i < textureDescs.size(); ++i) {
			if (strcmp(textureDescs[i].path.c_str(), path) == 0) {
				return textureDescs[i].id;
			}
		}

		TextureDesc res = { String{path}, static_cast<unsigned int>(textureDescs.size()) };
		textureDescs.push_back(res);

		texturesNeedUpdate = changesSinceLastCheck = true;

		return res.id;
	}

	const Material &getMaterial(MaterialId id) const {
		dassert(id >= 0 && id < materials.size() && id != INVALID_MATERIAL_ID);
		return materials[id];
	}

	const TextureDesc &getTextureDescription(TextureId id) const {
		dassert(id >= 0 && id < textures.size() && id != INVALID_TEXTURE_ID);
		return textureDescs[id];
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

	bool hadChangesSinceLastCheck() const {
		bool result = changesSinceLastCheck;
		changesSinceLastCheck = false;

		return result;
	}

	void prepareFrameData(Dar::FrameData &frameData);
	void prepareFrameDataForShadowMap(int shadowMapPassIndex, Dar::FrameData &frameData);

	LightId getLightcasterId(int lightcasterIndex) const;
	LightNode* getLightcaster(int lightcasterIndex) const;

private:
	bool uploadLightData(Dar::UploadHandle uploadHandle);
	bool uploadMaterialData(Dar::UploadHandle uploadHandle);

	// update view-projection matrices for moving lightcasters
	void updateLightData();
	void drawMeshes(Dar::FrameData &frameData);

	//void draw(Dar::FrameData &frameData) const;
	//void drawNodeImpl(Node *node, Dar::FrameData &frameData, const Scene &scene, DynamicBitset &drawnNodes) const;

private:
	Dar::HeapHandle texturesHeap; ///< Heap of the memory holding the textures' data

	bool texturesNeedUpdate; ///< Indicates textures have been changed and need to be reuploaded to the GPU.
	bool lightsNeedUpdate; ///< Indicates lights have been changed and need to be reuploaded to the GPU.
	bool materialsNeedUpdate; ///< Indicates materials have been changed and need to be reuploaded to the GPU.
	mutable bool changesSinceLastCheck; ///< Check if any changes in the textures/lights/materials was done since the last read of this value.

	mutable LightId lightcasterId = LightId(-1);
};
