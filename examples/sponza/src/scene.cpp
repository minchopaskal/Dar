#include "scene.h"

#include "utils/logger.h"
#include "utils/timer.h"

#include "reslib/img_data.h"
#include "reslib/resource_library.h"

void Mesh::uploadMeshData(Dar::UploadHandle uploadHandle) const {
	// Only upload the data if needed
	if (modelMatrix == cache && meshDataHandle != INVALID_RESOURCE_HANDLE) {
		return;
	}

	MeshData md = { modelMatrix, glm::transpose(glm::inverse(modelMatrix)), static_cast<unsigned int>(mat) };

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

void ModelNode::updateMeshDataHandles(const Scene &scene) const {
	Dar::ResourceManager &resManager = Dar::getResourceManager();
	Dar::UploadHandle handle = resManager.beginNewUpload();

	for (SizeType i = startMesh; i < startMesh + numMeshes; ++i) {
		scene.staticData.meshes[i].uploadMeshData(handle);
	}

	resManager.uploadBuffers();
}

void ModelNode::draw(Dar::FrameData &frameData, const Scene &scene) const {
	updateMeshDataHandles(scene);

	for (SizeType i = startMesh; i < startMesh + numMeshes; ++i) {
		const Mesh &mesh = scene.staticData.meshes[i];

		frameData.addRenderCommand(Dar::RenderCommandSetConstantBuffer(mesh.meshDataHandle, static_cast<UINT>(DefaultConstantBufferView::MeshData), false));
		frameData.addRenderCommand(Dar::RenderCommandDrawIndexedInstanced(static_cast<UINT>(mesh.numIndices), 1, static_cast<UINT>(mesh.indexOffset), 0, 0));
	}
}

Scene::Scene() :
	sceneBox(BBox::invalidBBox()),
	texturesNeedUpdate(true),
	lightsNeedUpdate(true),
	materialsNeedUpdate(true),
	changesSinceLastCheck(true) {
	for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
		lightcasterIndices[i] = -1;
	}
}

Scene::~Scene() {
	for (int i = 0; i < nodes.size(); ++i) {
		delete nodes[i];
	}

	nodes.clear();
}

bool Scene::uploadSceneData(Dar::UploadHandle uploadHandle) {
	LOG(Info, "Scene::uploadSceneData");

	if (!lightsNeedUpdate && !materialsNeedUpdate && !texturesNeedUpdate) {
		LOG(Info, "Scene::uploadSceneData SUCCESS");
		return true;
	}

	// TODO: try using placed resources for lights and materials OR small textures
	if (texturesNeedUpdate) {
		if (!uploadTextureData(textureDescs, uploadHandle, textures, texturesHeap, false)) {
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

	LOG(Info, "Scene::uploadSceneData SUCCESS");
	return true;
}

//void Scene::draw(Dar::FrameData &frameData) const {
//	const SizeType numNodes = nodes.size();
//	DynamicBitset drawnNodes(numNodes);
//	for (int i = 0; i < numNodes; ++i) {
//		drawNodeImpl(nodes[i], frameData, *this, drawnNodes);
//	}
//}
//
//void Scene::drawNodeImpl(Node *node, Dar::FrameData &frameData, const Scene &scene, DynamicBitset &drawnNodes) const {
//	if (drawnNodes[node->id]) {
//		return;
//	}
//
//	drawnNodes[node->id] = true;
//
//	node->draw(frameData, scene);
//
//	const SizeType numChildren = node->children.size();
//	for (int i = 0; i < numChildren; ++i) {
//		drawNodeImpl(nodes[node->children[i]], frameData, scene, drawnNodes);
//	}
//}

void Scene::prepareFrameData(Dar::FrameData &frameData, Dar::UploadHandle uploadHandle) {
	frameData.addDataBufferResource(materialsBuffer);
	for (int i = 0; i < textures.size(); ++i) {
		frameData.addTextureResource(textures[i]);
	}

	drawMeshes(frameData, uploadHandle);
}

void Scene::prepareFrameDataForAnimated(Dar::FrameData &frameData, Dar::UploadHandle uploadHandle) {
	frameData.addDataBufferResource(materialsBuffer);
	for (int i = 0; i < textures.size(); ++i) {
		frameData.addTextureResource(textures[i]);
	}

	drawAnimatedMeshes(frameData, uploadHandle);
}

void Scene::prepareFrameDataForShadowMap(int shadowMapPassIndex, Dar::FrameData & frameData, Dar::UploadHandle uploadHandle) {
	if (shadowMapPassIndex == 0) {
		updateLightData(uploadHandle);
	}

	if (getLightcaster(shadowMapPassIndex) == nullptr) {
		return;
	}

	// light buffer index in ResourceDescriptorHeap = 0. Really need to figure out a config working for cpu&gpu for these indices
	frameData.addDataBufferResource(lightsBuffer);

	// lightcasterDescs should already be uploaded in updateLightData()
	frameData.addRenderCommand(
		Dar::RenderCommandSetConstantBuffer(
			lightcasterDescs[shadowMapPassIndex].getHandle(),
			static_cast<int>(ShadowMapConstantBufferView::LightcasterDesc),
			false
		)
	);
	
	prepareFrameData(frameData, uploadHandle);
}

LightId Scene::getLightcasterId(int lightcasterIndex) const {
	return lightcasterIndices[lightcasterIndex];
}

LightNode* Scene::getLightcaster(int lightcasterIndex) const {
	auto id = getLightcasterId(lightcasterIndex);
	if (id < lightIndices.size()) {
		return dynamic_cast<LightNode *>(nodes[lightIndices[id]]);
	}

	return nullptr;
}

void Scene::updateLightData(Dar::UploadHandle uploadHandle) {
	for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
		lightcasterIndices[i] = -1;
	}

	int lcasterId = 0;
	for (int i = 0; i < lightIndices.size(); ++i) {
		auto light = dynamic_cast<LightNode *>(nodes[lightIndices[i]]);
		auto &gpuLight = light->lightData;
		gpuLight.shadowMapIndexOffset = -1;
		switch (gpuLight.type) {
		case LightType::Directional:
		{
			BoundingSphere s = sceneBox.getBoundingSphere();
			float diameter = 2.f * s.radius;
			Vec3 lightPos = s.center - s.radius * gpuLight.direction;

			auto projectionMatrix = glm::ortho(-s.radius, s.radius, -s.radius, s.radius, 0.1f, diameter);
			auto viewMatrix = glm::lookAt(lightPos, lightPos + gpuLight.direction, Vec3UnitY());

			gpuLight.viewProjection = projectionMatrix * viewMatrix;

			if (lcasterId < MAX_SHADOW_MAPS_COUNT) {
				gpuLight.shadowMapIndexOffset = lcasterId;
				lightcasterIndices[lcasterId++] = i;
			}

			break;
		}
		case LightType::Spot:
		{
			auto &cam = *getRenderCamera();

			Vec3 offset = 10.f * cam.getCameraZ() + 50.f * cam.getCameraY();

			// TODO: wiggle while moving?
			gpuLight.position = cam.getPos() + offset;
			gpuLight.direction = glm::normalize(cam.getPos() + cam.getCameraZ() * 200.f - gpuLight.position);
			gpuLight.zNear = 0.01f;
			gpuLight.zFar = 10000.f;
			Mat4 viewMatrix = glm::lookAt(gpuLight.position, gpuLight.position + gpuLight.direction, Vec3UnitY());
			Mat4 projectionMatrix = glm::perspective(glm::radians(cam.getFOV() + 30.f), cam.getAspectRatio(), 0.01f, 10000.f);

			gpuLight.viewProjection = projectionMatrix * viewMatrix;

			if (lcasterId < MAX_SHADOW_MAPS_COUNT) {
				gpuLight.shadowMapIndexOffset = lcasterId;
				lightcasterIndices[lcasterId++] = i;
			}

			break;
		}
		default:
			// TODO
			break;
		}
	}

	uploadLightData(uploadHandle);

	for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
		LightcasterDesc desc = {};
		desc.index = lightcasterIndices[i];

		lightcasterDescs[i].init(sizeof(LightcasterDesc), 1);
		lightcasterDescs[i].upload(uploadHandle, &desc);
	}
}

void Scene::drawMeshes(Dar::FrameData &frameData, Dar::UploadHandle uploadHandle) {
	for (SizeType i = 0; i < staticData.meshes.size(); ++i) {
		staticData.meshes[i].uploadMeshData(uploadHandle);
	}

	for (SizeType i = 0; i < staticData.meshes.size(); ++i) {
		const Mesh &mesh = staticData.meshes[i];

		frameData.addRenderCommand(Dar::RenderCommandSetConstantBuffer(mesh.meshDataHandle, static_cast<UINT>(DefaultConstantBufferView::MeshData), false));
		frameData.addRenderCommand(Dar::RenderCommandDrawIndexedInstanced(static_cast<UINT>(mesh.numIndices), 1, static_cast<UINT>(mesh.indexOffset), 0, 0));
	}
}

void Scene::drawAnimatedMeshes(Dar::FrameData &frameData, Dar::UploadHandle uploadHandle) {
	for (SizeType i = 0; i < animatedData.meshes.size(); ++i) {
		animatedData.meshes[i].uploadMeshData(uploadHandle);
	}

	for (SizeType i = 0; i < animatedData.meshes.size(); ++i) {
		const Mesh &mesh = animatedData.meshes[i];

		frameData.addRenderCommand(Dar::RenderCommandSetConstantBuffer(mesh.meshDataHandle, static_cast<UINT>(DefaultConstantBufferView::MeshData), false));
		frameData.addRenderCommand(Dar::RenderCommandDrawIndexedInstanced(static_cast<UINT>(mesh.numIndices), 1, static_cast<UINT>(mesh.indexOffset), 0, 0));
	}
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

		LightNode *light = dynamic_cast<LightNode *>(nodes[currLightIdx]);
		dassert(light);
		LightData &gpuLight = light->lightData;
		memcpy(lightsMemory + i * sizeof(LightData), &gpuLight, sizeof(LightData));
	}

	SizeType lightsOldBufferSize = lightsBuffer.getSize();
	if (lightsDataSize > lightsOldBufferSize) {
		lightsBuffer.init(sizeof(LightData), numLights);
	
		lightsBuffer.setName("LightsData");
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

	SizeType materialsOldBufferSize = materialsBuffer.getSize();
	if (materialsDataSize > materialsOldBufferSize) {
		materialsBuffer.init(sizeof(MaterialData), numMaterials);
		materialsBuffer.setName("MaterialsData");
	}

	if (!materialsBuffer.upload(uploadHandle, materialsMemory)) {
		LOG(Error, "Failed to material lights data!");
		return false;
	}

	free(materialsMemory);

	return true;
}
