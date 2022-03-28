#include "d3d12_scene.h"

#include "d3d12_asset_manager.h"

#include "d3d12_logger.h"

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
	const WString fullPathWStr = getAssetFullPath(imgPathWStr.c_str(), AssetType::Texture);
	const SizeType bufferLen = fullPathWStr.size() * sizeof(wchar_t) + 1;
	char *path = new char[bufferLen];
	stbi_convert_wchar_to_utf8(path, bufferLen, fullPathWStr.c_str());

	ImageData result = {};
	result.data = stbi_load(path, &result.width, &result.height, nullptr, 4);
	result.ncomp = 4;

	return result;
}

// TODO: move d3d12_scene to Sponza(future Renderer project).
// Then the MeshData will be taken from gpu_cpu_common.hlsli
struct MeshData {
	Mat4 modelMatrix;
	Mat4 normalMatrix;
	unsigned int materialId;
};

void Mesh::uploadMeshData(UploadHandle uploadHandle) const {
	// Only upload the data if needed
	if (modelMatrix == cache && meshDataHandle != INVALID_RESOURCE_HANDLE) {
		return;
	}

	MeshData md = { modelMatrix, modelMatrix.inverse().transpose(), static_cast<unsigned int>(mat) };
	
	ResourceManager &resManager = getResourceManager();
	if (meshDataHandle != INVALID_RESOURCE_HANDLE) {
		resManager.deregisterResource(meshDataHandle);
		meshDataHandle = INVALID_RESOURCE_HANDLE;
	}

	ResourceInitData resInit(ResourceType::DataBuffer);
	resInit.size = sizeof(MeshData);
	meshDataHandle = resManager.createBuffer(resInit);
	resManager.uploadBufferData(uploadHandle, meshDataHandle, &md, sizeof(MeshData));

	// update the cache for future uploads
	cache = modelMatrix;
}

void ModelNode::updateMeshDataHandles() const {
	ResourceManager &resManager = getResourceManager();
	UploadHandle handle = resManager.beginNewUpload();

	for (int i = 0; i < meshes.size(); ++i) {
		meshes[i].uploadMeshData(handle);
	}

	resManager.uploadBuffers();
}

void ModelNode::draw(CommandList &cmdList, const Scene &scene) const {
	const SizeType numMeshes = meshes.size();

	updateMeshDataHandles();

	for (int i = 0; i < numMeshes; ++i) {
		const Mesh &mesh = meshes[i];

		cmdList.transition(mesh.meshDataHandle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		cmdList.setConstantBufferView(static_cast<unsigned int>(ConstantBufferView::MeshData), static_cast<unsigned int>(mesh.meshDataHandle));
		cmdList->DrawIndexedInstanced(static_cast<UINT>(mesh.numIndices), 1, static_cast<UINT>(mesh.indexOffset), 0, 0);
	}
}

Scene::Scene(ComPtr<ID3D12Device8> &device) :
	sceneBox(BBox::invalidBBox()),
	device(device),
	texturesNeedUpdate(true),
	lightsNeedUpdate(true),
	materialsNeedUpdate(true),
	changesSinceLastCheck(true)
{ }

Scene::~Scene() {
	for (int i = 0; i < nodes.size(); ++i) {
		delete nodes[i];
	}

	nodes.clear();
}

bool Scene::uploadSceneData(UploadHandle uploadHandle) {
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

void Scene::draw(CommandList &cmdList) const {
	const SizeType numNodes = nodes.size();
	DynamicBitset drawnNodes(numNodes);
	for (int i = 0; i < numNodes; ++i) {
		drawNodeImpl(nodes[i], cmdList, *this, drawnNodes);
	}
}

void Scene::drawNodeImpl(Node *node, CommandList &cmdList, const Scene &scene, DynamicBitset &drawnNodes) const {
	if (drawnNodes[node->id]) {
		return;
	}

	drawnNodes[node->id] = true;

	node->draw(cmdList, scene);

	const SizeType numChildren = node->children.size();
	for (int i = 0; i < numChildren; ++i) {
		drawNodeImpl(nodes[node->children[i]], cmdList, scene, drawnNodes);
	}
}

bool Scene::uploadLightData(UploadHandle uploadHandle) {
	SizeType numLights = getNumLights();
	if (numLights == 0) {
		return true;
	}

	SizeType lightsDataSize = numLights * sizeof(GPULight);
	Byte *lightsMemory = (Byte*)malloc(lightsDataSize);
	if (lightsMemory == nullptr) {
		return false;
	}

	for (int i = 0; i < numLights; ++i) {
		LightId currLightIdx = lightIndices[i];
		dassert(nodes[currLightIdx]->getNodeType() == NodeType::Light);

		LightNode *light = dynamic_cast<LightNode*>(nodes[currLightIdx]);
		if (light) {
			// Do as much preprocessing as possible
			GPULight gpuLight = {};
			gpuLight.ambient = light->ambient;
			gpuLight.attenuation = light->attenuation;
			gpuLight.diffuse = light->diffuse;
			gpuLight.direction = dmath::normalized(light->direction);
			gpuLight.innerAngleCutoff = cos(light->innerAngleCutoff);
			gpuLight.outerAngleCutoff = cos(light->outerAngleCutoff);
			gpuLight.position = light->position;
			gpuLight.specular = light->specular;
			gpuLight.type = static_cast<int>(light->type);
			memcpy(lightsMemory + i * sizeof(GPULight), &gpuLight, sizeof(GPULight));
		}
	}

	ResourceManager &resManager = getResourceManager();

	ResourceInitData resData(ResourceType::DataBuffer);
	resData.size = lightsDataSize;
	resData.name = L"LightsData";
	lightsHandle = resManager.createBuffer(resData);

	if (!resManager.uploadBufferData(uploadHandle, lightsHandle, lightsMemory, lightsDataSize)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to upload lights data!");
		return false;
	}

	free(lightsMemory);
	return true;
}

bool Scene::uploadMaterialData(UploadHandle uploadHandle) {
	SizeType numMaterials = getNumMaterials();
	if (numMaterials == 0) {
		return true;
	}

	SizeType materialsDataSize = numMaterials * sizeof(MaterialData);
	Byte *materialsMemory = ( Byte* )malloc(materialsDataSize);
	if (materialsMemory == nullptr) {
		return false;
	}

	for (int i = 0; i < numMaterials; ++i) {
		const Material &m = getMaterial(i);
		memcpy(materialsMemory + i * sizeof(MaterialData), &m.materialData, sizeof(MaterialData));
	}

	ResourceManager &resManager = getResourceManager();

	ResourceInitData resData(ResourceType::DataBuffer);
	resData.size = materialsDataSize;
	resData.name = L"MaterialsData";
	materialsHandle = resManager.createBuffer(resData);

	if (!resManager.uploadBufferData(uploadHandle, materialsHandle, materialsMemory, materialsDataSize)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to material lights data!");
		return false;
	}

	free(materialsMemory);

	return true;
}

bool Scene::uploadTextureData(UploadHandle uploadHandle) {
	ResourceManager &resManager = getResourceManager();

	SizeType numTextures = getNumTextures();

	Vector<ImageData> texData(numTextures);
	Vector<ComPtr<ID3D12Resource>> stagingImageBuffers(numTextures);
	for (int i = 0; i < numTextures; ++i) {
		if (textureHandles[i] != INVALID_RESOURCE_HANDLE) {
			continue;
		}

		Texture &tex = textures[i];
		texData[i] = loadImage(tex.path);
		tex.format = texData[i].ncomp == 4 ? TextureFormat::RGBA_8BIT : TextureFormat::Invalid;
		wchar_t textureName[32] = L"";
		swprintf(textureName, 32, L"Texture[%d]", i);

		ResourceInitData texInitData(ResourceType::TextureBuffer);
		texInitData.textureData.width = texData[i].width;
		texInitData.textureData.height = texData[i].height;
		texInitData.textureData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
		texInitData.name = textureName;
		textureHandles[i] = resManager.createBuffer(texInitData);
		if (textureHandles[i] == INVALID_RESOURCE_HANDLE) {
			return false;
		}
	}

	for (int i = 0; i < numTextures; ++i) {
		D3D12_SUBRESOURCE_DATA textureSubresources = {};
		textureSubresources.pData = texData[i].data;
		textureSubresources.RowPitch = static_cast< UINT64 >(texData[i].width) * static_cast< UINT64 >(texData[i].ncomp);
		textureSubresources.SlicePitch = textureSubresources.RowPitch * texData[i].height;
		resManager.uploadTextureData(uploadHandle, textureHandles[i], &textureSubresources, 1, 0);
	}

	return true;
}
