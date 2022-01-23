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
	materialsNeedUpdate(true)
{ }

Scene::~Scene() {
	for (int i = 0; i < nodes.size(); ++i) {
		delete nodes[i];
	}

	nodes.clear();
}

ID3D12DescriptorHeap* const* Scene::getSrvHeap() {
	return srvHeap.GetAddressOf();
}

bool Scene::uploadSceneData() {
	if (!lightsNeedUpdate && !materialsNeedUpdate && !texturesNeedUpdate) {
		return true;
	}

	ResourceManager &resManager = getResourceManager();
	UploadHandle handle = resManager.beginNewUpload();

	// TODO: try using placed resources for lights and materials OR small textures
	if (texturesNeedUpdate) {
		if (!uploadTextureData(handle)) {
			LOG(Error, "Failed to upload texture data!");
			return false;
		}
	}

	if (lightsNeedUpdate) {
		if (!uploadLightData(handle)) {
			LOG(Error, "Failed to upload light data!");
			return false;
		}
	}

	if (materialsNeedUpdate) {
		if (!uploadMaterialData(handle)) {
			LOG(Error, "Failed to upload material data!");
			return false;
		}
	}

	texturesNeedUpdate = lightsNeedUpdate = materialsNeedUpdate = false;

	resManager.uploadBuffers();

	/* Create shader resource view heap which will store the handles to the textures */
	srvHeap.Reset();

	const UINT numTextures = static_cast<UINT>(getNumTextures());
	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.NumDescriptors = numTextures + 2;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	srvHeapDesc.NodeMask = 0;
	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap)),
		"Failed to create DSV descriptor heap!"
	);

	SizeType srvHeapHandleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle = srvHeap->GetCPUDescriptorHandleForHeapStart();

	// Create SRV for the lights
	if (getNumLights() > 0) {
		D3D12_SHADER_RESOURCE_VIEW_DESC lightsSrvDesc = {};
		lightsSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		lightsSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		lightsSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		lightsSrvDesc.Buffer = {};
		lightsSrvDesc.Buffer.FirstElement = 0;
		lightsSrvDesc.Buffer.NumElements = getNumLights();
		lightsSrvDesc.Buffer.StructureByteStride = sizeof(GPULight);
		device->CreateShaderResourceView(lightsHandle.get(), &lightsSrvDesc, descriptorHandle);
	}
	descriptorHandle.ptr += srvHeapHandleSize;

	// Create SRV for the materials
	if (getNumMaterials() > 0) {
		D3D12_SHADER_RESOURCE_VIEW_DESC materialsSrvDesc = {};
		materialsSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		materialsSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		materialsSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		materialsSrvDesc.Buffer = {};
		materialsSrvDesc.Buffer.FirstElement = 0;
		materialsSrvDesc.Buffer.NumElements = getNumMaterials();
		materialsSrvDesc.Buffer.StructureByteStride = sizeof(GPUMaterial);
		device->CreateShaderResourceView(materialsHandle.get(), &materialsSrvDesc, descriptorHandle);
	}
	descriptorHandle.ptr += srvHeapHandleSize;

	/* Create SRVs for the textures so we can read them bindlessly in the shader */
	for (int i = 0; i < numTextures; ++i) {
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = getTexture(i).format == TextureFormat::RGBA_8BIT ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Texture2D.MipLevels = 1;

		device->CreateShaderResourceView(textureHandles[i].get(), &srvDesc, descriptorHandle);
		descriptorHandle.ptr += srvHeapHandleSize;
	}
}

void Scene::draw(CommandList &cmdList) const {
	// TODO: any lights/cameras/other objects global to the scene
	// that need to go as data in the shader, should go here
	// before recursing the node tree.

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

	SizeType materialsDataSize = numMaterials * sizeof(GPUMaterial);
	Byte *materialsMemory = ( Byte* )malloc(materialsDataSize);
	if (materialsMemory == nullptr) {
		return false;
	}

	for (int i = 0; i < numMaterials; ++i) {
		const Material &m = getMaterial(i);
		GPUMaterial gpuM = { m.diffuse, m.specular, m.normals };
		memcpy(materialsMemory + i * sizeof(GPUMaterial), &gpuM, sizeof(GPUMaterial));
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
	srvHeap.Reset();

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
}
