#include "d3d12_scene.h"

#include "d3d12_resource_manager.h"

void Model::draw(CommandList &cmdList, const Scene &scene) const {
	const SizeType numMeshes = meshes.size();
	for (int i = 0; i < numMeshes; ++i) {
		const Mesh &mesh = meshes[i];

		MaterialId matId = mesh.mat;
		if (matId != INVALID_MATERIAL_ID) {
			ResourceHandle matHandle = scene.getMaterialHandle(matId);
			if (matHandle != INVALID_RESOURCE_HANDLE) {
				cmdList->SetGraphicsRootConstantBufferView(1, matHandle->GetGPUVirtualAddress());
			}
		}

		cmdList->DrawIndexedInstanced(static_cast<UINT>(mesh.numIndices), 1, static_cast<UINT>(mesh.indexOffset), 0, 0);
	}
}

Scene::Scene() : sceneBox(BBox::invalidBBox()) { }

void Scene::draw(CommandList &cmdList) const {
	// TODO: any lights/cameras/other objects global to the scene
	// that need to go as data in the shader, should go here
	// before recursing the node tree.

	const SizeType numNodes = nodes.size();
	for (int i = 0; i < numNodes; ++i) {
		drawNodeImpl(nodes[i], cmdList, *this);
	}
}

void Scene::drawNodeImpl(Node *node, CommandList &cmdList, const Scene &scene) const {
	node->draw(cmdList, scene);

	const SizeType numChildren = node->children.size();
	for (int i = 0; i < numChildren; ++i) {
		drawNodeImpl(nodes[node->children[i]], cmdList, scene);
	}
}

struct D3D12Material {
	unsigned int diffuseIdx;
	unsigned int specularIdx;
	unsigned int normalsIdx;
};

void Scene::uploadMaterialBuffers() {
	dassert(materials.size() == materialHandles.size());

	// Set-up step to upload any materials, which were not yet
	// intialized on the GPU
	ResourceManager &resManager = getResourceManager();

	UploadHandle uploadHandle = resManager.beginNewUpload();
	const SizeType numMaterials = getNumMaterials();
	for (unsigned int i = 0; i < numMaterials; ++i) {
		const Material &m = getMaterial(i);
		if (materialHandles[i] != INVALID_RESOURCE_HANDLE) {
			continue;
		}
		
		D3D12Material d12Mat = { static_cast<UINT>(m.diffuse), static_cast<UINT>(m.specular), static_cast<UINT>(m.normals) };
		wchar_t materialName[32] = L"";
		swprintf(materialName, 32, L"Material[%u]", i);

		ResourceInitData resData(ResourceType::DataBuffer);
		resData.size = sizeof(D3D12Material);
		resData.name = materialName;
		materialHandles[i] = resManager.createBuffer(resData);
		resManager.uploadBufferData(uploadHandle, materialHandles[i], reinterpret_cast<void*>(&d12Mat), sizeof(D3D12Material));
	}

	resManager.uploadBuffers();
}

Mat4 Camera::getViewMatrix() const {
	// TODO: is this correct?
	return Mat4{ orientation.getRotationMatrix(), Vec4(pos, 1.f) };
}

void Camera::move(const Vec3 &magnitude) {
	pos = pos + magnitude;
}

void Camera::rotate(const Vec3 &axis, float radians) {

}

void Camera::zoom(float factor) {
	switch (type) {
	case CameraType::Perspective:
		break;
	case CameraType::Orthographic:
		// TODO;
		break;
	default:
		dassert(false);
		break;
	}
}

Camera&& Camera::perspectiveCamera(const Vec3 &pos, float fov, float aspectRatio, float nearPlane, float farPlane) {
	Camera res;

	res.pos = pos;
	res.fov = fov;
	res.aspectRatio = aspectRatio;
	res.nearPlane = nearPlane;
	res.farPlane = farPlane;
	res.type = CameraType::Perspective;

	return std::move(res);
}

Camera&& Camera::orthographicCamera(const Vec3 &pos, float renderRectWidth, float renderRectHeight, float nearPlane, float farPlane) {
	Camera res;

	res.pos = pos;
	res.width = renderRectWidth;
	res.height = renderRectHeight;
	res.nearPlane = nearPlane;
	res.farPlane = farPlane;
	res.type = CameraType::Orthographic;

	return std::move(res);
}

Mat4 Camera::getProjectionMatrix() const {
	switch (type) {
	case CameraType::Perspective:
		return dmath::perspective(fov, aspectRatio, nearPlane, farPlane);
	case CameraType::Orthographic:
		return dmath::orthographic(-(width / 2), width / 2, -(height / 2), height / 2, nearPlane, farPlane);
	default:
		dassert(false);
		return Mat4(1.f);
	}
}
