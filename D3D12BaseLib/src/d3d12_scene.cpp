#include "d3d12_scene.h"

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

Scene::Scene() : 
	sceneBox(BBox::invalidBBox()),
	materialsHandle(INVALID_RESOURCE_HANDLE),
	lightsHandle(INVALID_RESOURCE_HANDLE),
	lightsNeedUpdate(true),
	materialsNeedUpdate(true)
{ }

Scene::~Scene() {
	for (int i = 0; i < nodes.size(); ++i) {
		delete nodes[i];
	}

	nodes.clear();
}

void Scene::uploadSceneData() {
	if (!lightsNeedUpdate && !materialsNeedUpdate) {
		return;
	}

	ResourceManager &resManager = getResourceManager();
	UploadHandle handle = resManager.beginNewUpload();

	if (lightsNeedUpdate) {
		uploadLightData(handle);
	}

	if (materialsNeedUpdate) {
		uploadMaterialData(handle);
	}

	resManager.uploadBuffers();
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

void Scene::uploadLightData(UploadHandle uploadHandle) {
	SizeType numLights = getNumLights();
	if (numLights == 0) {
		return;
	}

	SizeType lightsDataSize = numLights * sizeof(GPULight);
	Byte *lightsMemory = (Byte*)malloc(lightsDataSize);
	if (lightsMemory == nullptr) {
		return;
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

	resManager.uploadBufferData(uploadHandle, lightsHandle, lightsMemory, lightsDataSize);

	free(lightsMemory);
}

void Scene::uploadMaterialData(UploadHandle uploadHandle) {
	SizeType numMaterials = getNumMaterials();
	if (numMaterials == 0) {
		return;
	}

	SizeType materialsDataSize = numMaterials * sizeof(GPUMaterial);
	Byte *materialsMemory = ( Byte* )malloc(materialsDataSize);
	if (materialsMemory == nullptr) {
		return;
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

	resManager.uploadBufferData(uploadHandle, materialsHandle, materialsMemory, materialsDataSize);

	free(materialsMemory);
}
