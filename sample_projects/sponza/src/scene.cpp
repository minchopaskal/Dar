#include "scene.h"

#include "asset_manager/asset_manager.h"
#include "utils/logger.h"
#include "utils/timer.h"

// For loading the texture image
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

struct ImageData {
	void *data = nullptr;
	int width = 0;
	int height = 0;
	int ncomp = 0;
};

ImageData loadImage(const String &imgPath) {
	WString imgPathWStr(imgPath.begin(), imgPath.end());
	const WString fullPathWStr = Dar::getAssetFullPath(imgPathWStr.c_str(), Dar::AssetType::Texture);
	const SizeType bufferLen = fullPathWStr.size() * sizeof(wchar_t) + 1;
	char *path = new char[bufferLen];
	stbi_convert_wchar_to_utf8(path, bufferLen, fullPathWStr.c_str());

	ImageData result = {};
	result.data = stbi_load(path, &result.width, &result.height, nullptr, 4);
	result.ncomp = 4;

	return result;
}

void Mesh::uploadMeshData(Dar::UploadHandle uploadHandle) const {
	// Only upload the data if needed
	if (modelMatrix == cache && meshDataHandle != INVALID_RESOURCE_HANDLE) {
		return;
	}

	MeshData md = { modelMatrix, modelMatrix.inverse().transpose(), static_cast<unsigned int>(mat) };

	Dar::ResourceManager &resManager = Dar::getResourceManager();
	if (meshDataHandle != INVALID_RESOURCE_HANDLE) {
		resManager.deregisterResource(meshDataHandle);
		meshDataHandle = INVALID_RESOURCE_HANDLE;
	}

	Dar::ResourceInitData resInit(Dar::ResourceType::DataBuffer);
	resInit.size = sizeof(MeshData);
	meshDataHandle = resManager.createBuffer(resInit);
	resManager.uploadBufferData(uploadHandle, meshDataHandle, &md, sizeof(MeshData));

	// update the cache for future uploads
	cache = modelMatrix;
}

void ModelNode::updateMeshDataHandles() const {
	Dar::ResourceManager &resManager = Dar::getResourceManager();
	Dar::UploadHandle handle = resManager.beginNewUpload();

	for (int i = 0; i < meshes.size(); ++i) {
		meshes[i].uploadMeshData(handle);
	}

	resManager.uploadBuffers();
}

void ModelNode::draw(Dar::FrameData &frameData, const Scene &scene) const {
	const SizeType numMeshes = meshes.size();

	updateMeshDataHandles();

	for (int i = 0; i < numMeshes; ++i) {
		const Mesh &mesh = meshes[i];

		frameData.addRenderCommand(Dar::RenderCommandSetConstantBuffer(mesh.meshDataHandle, static_cast<UINT>(ConstantBufferView::MeshData)), 0);
		frameData.addRenderCommand(Dar::RenderCommandDrawIndexedInstanced(static_cast<UINT>(mesh.numIndices), 1, static_cast<UINT>(mesh.indexOffset), 0, 0), 0);
	}
}

Scene::Scene() :
	sceneBox(BBox::invalidBBox()),
	texturesNeedUpdate(true),
	lightsNeedUpdate(true),
	materialsNeedUpdate(true),
	changesSinceLastCheck(true) {}

Scene::~Scene() {
	for (int i = 0; i < nodes.size(); ++i) {
		delete nodes[i];
	}

	nodes.clear();
}

bool Scene::uploadSceneData(Dar::UploadHandle uploadHandle) {
	if (!lightsNeedUpdate && !materialsNeedUpdate && !texturesNeedUpdate) {
		return true;
	}

	// TODO: try using placed resources for lights and materials OR small textures
	if (texturesNeedUpdate) {
		if (!uploadTextureData(uploadHandle)) {
			LOG(Error, "Failed to upload texture data!");
			return false;
		}
	}

	if (lightsNeedUpdate) {
		if (!uploadLightData(uploadHandle)) {
			LOG(Error, "Failed to upload light data!");
			return false;
		}
	}

	if (materialsNeedUpdate) {
		if (!uploadMaterialData(uploadHandle)) {
			LOG(Error, "Failed to upload material data!");
			return false;
		}
	}

	texturesNeedUpdate = lightsNeedUpdate = materialsNeedUpdate = false;

	return true;
}

void Scene::draw(Dar::FrameData &frameData) const {
	const SizeType numNodes = nodes.size();
	DynamicBitset drawnNodes(numNodes);
	for (int i = 0; i < numNodes; ++i) {
		drawNodeImpl(nodes[i], frameData, *this, drawnNodes);
	}
}

void Scene::drawNodeImpl(Node *node, Dar::FrameData &frameData, const Scene &scene, DynamicBitset &drawnNodes) const {
	if (drawnNodes[node->id]) {
		return;
	}

	drawnNodes[node->id] = true;

	node->draw(frameData, scene);

	const SizeType numChildren = node->children.size();
	for (int i = 0; i < numChildren; ++i) {
		drawNodeImpl(nodes[node->children[i]], frameData, scene, drawnNodes);
	}
}

void Scene::prepareFrameData(Dar::FrameData &frameData) {
	frameData.addDataBufferResource(materialsBuffer, 0);
	for (int i = 0; i < textures.size(); ++i) {
		frameData.addTextureResource(textures[i], 0);
	}

	draw(frameData);
}

bool Scene::uploadLightData(Dar::UploadHandle uploadHandle) {
	SizeType numLights = getNumLights();
	if (numLights == 0) {
		return true;
	}

	SizeType lightsDataSize = numLights * sizeof(LightData);
	Byte *lightsMemory = (Byte *)malloc(lightsDataSize);
	if (lightsMemory == nullptr) {
		return false;
	}

	for (int i = 0; i < numLights; ++i) {
		LightId currLightIdx = lightIndices[i];
		dassert(nodes[currLightIdx]->getNodeType() == NodeType::Light);

		LightNode *light = dynamic_cast<LightNode *>(nodes[currLightIdx]);
		if (light) {
			// Do as much preprocessing as possible
			LightData &gpuLight = light->lightData;
			gpuLight.direction = dmath::normalized(gpuLight.direction);
			gpuLight.innerAngleCutoff = cos(gpuLight.innerAngleCutoff);
			gpuLight.outerAngleCutoff = cos(gpuLight.outerAngleCutoff);
			memcpy(lightsMemory + i * sizeof(LightData), &gpuLight, sizeof(LightData));
		}
	}

	Dar::ResourceManager &resManager = Dar::getResourceManager();

	SizeType lightsOldBufferSize = lightsBuffer.getSize();
	if (lightsDataSize > lightsOldBufferSize) {
		lightsBuffer.init(sizeof(LightData), lightsDataSize);
	
		lightsBuffer.setName(L"LightsData");
	}

	if (!lightsBuffer.upload(uploadHandle, lightsMemory)) {
		LOG(Error, "Failed to upload lights data!");
		return false;
	}

	free(lightsMemory);
	return true;
}

bool Scene::uploadMaterialData(Dar::UploadHandle uploadHandle) {
	SizeType numMaterials = getNumMaterials();
	if (numMaterials == 0) {
		return true;
	}

	SizeType materialsDataSize = numMaterials * sizeof(MaterialData);
	Byte *materialsMemory = (Byte *)malloc(materialsDataSize);
	if (materialsMemory == nullptr) {
		return false;
	}

	for (int i = 0; i < numMaterials; ++i) {
		const Material &m = getMaterial(i);
		memcpy(materialsMemory + i * sizeof(MaterialData), &m.materialData, sizeof(MaterialData));
	}

	Dar::ResourceManager &resManager = Dar::getResourceManager();

	SizeType materialsOldBufferSize = materialsBuffer.getSize();
	if (materialsDataSize > materialsOldBufferSize) {
		materialsBuffer.init(sizeof(MaterialData), numMaterials);
		materialsBuffer.setName(L"MaterialsData");
	}

	if (!materialsBuffer.upload(uploadHandle, materialsMemory)) {
		LOG(Error, "Failed to material lights data!");
		return false;
	}

	free(materialsMemory);

	return true;
}

bool Scene::uploadTextureData(Dar::UploadHandle uploadHandle) {
	Dar::ResourceManager &resManager = Dar::getResourceManager();

	SizeType numTextures = getNumTextures();

	std::for_each(
		textures.begin(),
		textures.end(),
		[](Dar::TextureResource &texResource) {
			texResource.deinit();
		}
	);

	textures.resize(numTextures);

	Vector<ImageData> texData(numTextures);
	Vector<D3D12_RESOURCE_DESC> texDescs(numTextures);
	Vector<Dar::TextureInitData> texInitDatas(numTextures);
	for (int i = 0; i < numTextures; ++i) {
		TextureDesc &tex = textureDescs[i];
		texData[i] = loadImage(tex.path);
		tex.format = texData[i].ncomp == 4 ? TextureFormat::RGBA_8BIT : TextureFormat::Invalid;

		wchar_t textureName[32] = L"";
		swprintf(textureName, 32, L"Texture[%d]", i);

		Dar::TextureInitData &texInitData = texInitDatas[i];
		texInitData.width = texData[i].width;
		texInitData.height = texData[i].height;
		texInitData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
		
		Dar::ResourceInitData resInitData = {};
		resInitData.init(Dar::ResourceType::TextureBuffer);
		resInitData.textureData = texInitData;
		resInitData.name = textureName;

		texDescs[i] = resInitData.getResourceDescriptor();
	}

	resManager.createHeap(texDescs.data(), static_cast<UINT>(texDescs.size()), texturesHeap);

	if (texturesHeap == INVALID_HEAP_HANDLE) {
		return false;
	}

	SizeType heapOffset = 0;
	for (int i = 0; i < numTextures; ++i) {
		Dar::HeapInfo heapInfo = {};
		heapInfo.handle = texturesHeap;
		heapInfo.offset = heapOffset;

		Dar::TextureInitData initData = {};
		initData.width = texData[i].width;
		initData.height = texData[i].height;
		initData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
		textures[i].init(initData, Dar::TextureResourceType::ShaderResource, &heapInfo);

		UINT64 size = textures[i].upload(uploadHandle, texData[i].data);

		heapOffset += size;
	}

	return true;
}
