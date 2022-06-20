#pragma once

#include "framework/camera.h"
#include "d3d12/command_list.h"
#include "d3d12/descriptor_heap.h"
#include "d3d12/resource_manager.h"
#include "graphics/renderer.h"
#include "math/dar_math.h"
#include "utils/defines.h"

#include "animation.h"
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
	String path = "";
	TextureId id = INVALID_TEXTURE_ID;
	TextureType type = TextureType::Invalid;
	TextureFormat format = TextureFormat::Invalid;
};

enum class MeshType {
	Static,
	Skinned
};

struct Mesh {
	Mat4 modelMatrix = Mat4(1.f);
	mutable Dar::ResourceHandle meshDataHandle = INVALID_RESOURCE_HANDLE;
	MaterialId mat = INVALID_MATERIAL_ID;
	SizeType indexOffset = 0;
	SizeType numIndices = 0;
	MeshType type;

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
	Vector<Mesh> meshes;

	ModelNode() {
		nodeType = NodeType::Model;
	}

	virtual void draw(Dar::FrameData &frameData, const Scene &scene) const override;

protected:
	void updateMeshDataHandles() const;
};

struct SkinnedModelNode : ModelNode {
	SkeletonId skeleton;
	Vector<AnimationId> animations;
	int currentAnimation;

	virtual void draw(Dar::FrameData &frameData, const Scene &scene) const override;
};

struct LightNode : Node {
	LightData lightData;

	LightNode() {
		lightData.type = LightType::Invalid;
		nodeType = NodeType::Light;
	}

	void draw(Dar::FrameData &frameData, const Scene &scene) const override final {
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

	virtual void draw(Dar::FrameData &frameData, const Scene &) const override final {}

	Dar::Camera *getCamera() {
		return &camera;
	}

private:
	Dar::Camera camera;
};

struct StaticVertex {
	Vec3 pos;
	Vec3 normal;
	Vec3 tangent;
	Vec2 uv;
};

#define MAX_VERTEX_BONES 4
struct SkinnedVertex : StaticVertex {
	Vec4u8 boneIDs; ///< We have 4 bones so 4 bone indices. We allow maximum of 256 bones per mesh
	// TODO: GEA suggests compression for the weights here.
	Vec3 boneWeights; ///< We have 4 bones, but weights are normalized so the 4th weight is just 1 - sum(boneWeights[i]{i=0,1,2});
};

// TODO: encapsulate members
struct Scene {
	template <class VertexType>
	struct VertexData {
		Vector<VertexType> vertices; ///< All vertices in the scene.
		Vector<unsigned int> indices; ///< All indices for all meshes, indexing in the vertices array.
	};

	Vector<Node*> nodes; ///< Vector with pointers to all nodes in the scene
	Vector<LightId> lightIndices; ///< Indices of the lights in the nodes vector
	Vector<CameraId> cameraIndices; ///< Indices of the cameras in the nodes vector
	Vector<Material> materials; ///< Vector with all materials in the scene
	Vector<TextureDesc> textureDescs; ///< Vector with all textures in the scene. Meshes could share texture ids.
	VertexData<StaticVertex> staticData; ///< Static scene data. These are all meshes that are not skinned.
	VertexData<SkinnedVertex> movingData; ///< Non-static data. These are all skinned meshes.
	Vector<Dar::TextureResource> textures; ///< Dx12 Texture resources
	AnimationManager animationManager;
	Dar::DataBufferResource materialsBuffer; ///< GPU buffer holding all materials' data.
	Dar::DataBufferResource lightsBuffer; ///< GPU buffer holding all lights' data.
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
		for (int i = 0; i < textureDescs.size(); ++i) {
			if (strcmp(textureDescs[i].path.c_str(), path) == 0) {
				return textureDescs[i].id;
			}
		}

		TextureDesc res = { String{path}, static_cast<unsigned int>(textures.size()), type };
		textureDescs.push_back(res);
		textures.push_back({});

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

	const void *getStaticVertexBuffer() const {
		return &staticData.vertices[0];
	}

	const void *getStaticIndexBuffer() const {
		return &staticData.indices[0];
	}

	const UINT getStaticVertexBufferSize() const {
		SizeType sz = staticData.vertices.size() == 0 ? 0 : staticData.vertices.size() * sizeof(staticData.vertices[0]);
		dassert(sz < ((SizeType(1) << 32) - 1));
		return static_cast<UINT>(sz);
	}

	const UINT getStaticIndexBufferSize() const {
		SizeType sz = staticData.indices.size() == 0 ? 0 : staticData.indices.size() * sizeof(staticData.indices[0]);
		dassert(sz < ((SizeType(1) << 32) - 1));
		return static_cast<UINT>(sz);
	}

	const void *getMovingVertexBuffer() const {
		return &movingData.vertices[0];
	}

	const void *getMovingIndexBuffer() const {
		return &movingData.indices[0];
	}

	const UINT getMovingVertexBufferSize() const {
		SizeType sz = movingData.vertices.size() == 0 ? 0 : movingData.vertices.size() * sizeof(movingData.vertices[0]);
		dassert(sz < ((SizeType(1) << 32) - 1));
		return static_cast<UINT>(sz);
	}

	const UINT getMovingIndexBufferSize() const {
		SizeType sz = movingData.indices.size() == 0 ? 0 : movingData.indices.size() * sizeof(movingData.indices[0]);
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

	void draw(Dar::FrameData &frameData) const;

	bool hadChangesSinceLastCheck() const {
		bool result = changesSinceLastCheck;
		changesSinceLastCheck = false;

		return result;
	}

	void prepareFrameData(Dar::FrameData &frameData);

	SkeletonId addNewSkeleton() {
		animationManager.skeletons.emplace_back();
		return animationManager.skeletons.size() - 1;
	}

	AnimationId addNewAnimation(AnimationClip &a) {
		animationManager.animations.push_back(a);
		return animationManager.animations.size() - 1;
	}

	AnimationId addNewAnimationClip() {
		animationManager.animations.emplace_back();
		return animationManager.animations.size() - 1;
	}

	AnimationClip& getAnimationClip(AnimationId id) {
		return animationManager.animations[id];
	}

	AnimationSkeleton& getSkeleton(SkeletonId id) {
		dassert(id < animationManager.skeletons.size());

		return animationManager.skeletons[id];
	}

	const AnimationSkeleton& getSkeleton(SkeletonId id) const {
		dassert(id < animationManager.skeletons.size());
		return animationManager.skeletons[id];
	}

private:
	bool uploadLightData(Dar::UploadHandle uploadHandle);
	bool uploadMaterialData(Dar::UploadHandle uploadHandle);
	bool uploadTextureData(Dar::UploadHandle uploadHandle);

	void drawNodeImpl(Node *node, Dar::FrameData &frameData, const Scene &scene, DynamicBitset &drawnNodes) const;

private:
	Dar::HeapHandle texturesHeap; ///< Heap of the memory holding the textures' data

	bool texturesNeedUpdate; ///< Indicates textures have been changed and need to be reuploaded to the GPU.
	bool lightsNeedUpdate; ///< Indicates lights have been changed and need to be reuploaded to the GPU.
	bool materialsNeedUpdate; ///< Indicates materials have been changed and need to be reuploaded to the GPU.
	mutable bool changesSinceLastCheck; ///< Check if any changes in the textures/lights/materials was done since the last read of this value.
};
