#include "scene_loader.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

// !!! Note: not thread-safe.
static Assimp::Importer *importer = nullptr;
Assimp::Importer& getAssimpImporter() {
	if (importer == nullptr) {
		importer = new Assimp::Importer;
	}

	return *importer;
}

TextureId loadTexture(aiMaterial *aiMat, aiTextureType aiType, TextureType matType, Scene &scene) {
	const int matTypeCount = aiMat->GetTextureCount(aiType);
	if (matTypeCount <= 0) {
		return INVALID_TEXTURE_ID;
	}

	// Only read the first texture, for now we won't export
	// multiple textures per channel.
	aiString path;
	aiMat->GetTexture(aiType, 0, &path);

	if (path.length <= 0) {
		return INVALID_TEXTURE_ID;
	}

	// Do not support embeded textures, yet.
	if (*path.C_Str() == '*') {
		dassert(false);
		return INVALID_TEXTURE_ID;
	}

	return scene.getNewTexture(path.C_Str(), matType);
}

MaterialId readMaterialDataForMesh(aiMesh *mesh, const aiScene *sc, Scene &scene) {
	auto matIdx = mesh->mMaterialIndex;
	if (matIdx < 0) {
		return INVALID_MATERIAL_ID;
	}

	aiMaterial *aiMat = sc->mMaterials[matIdx];

	TextureId diffuse = loadTexture(aiMat, aiTextureType_DIFFUSE, TextureType::Diffuse, scene);
	TextureId specular = loadTexture(aiMat, aiTextureType_SPECULAR, TextureType::Specular, scene);
	TextureId normals = loadTexture(aiMat, aiTextureType_NORMALS, TextureType::Normals, scene);

	return scene.getNewMaterial(diffuse, specular, normals);
}

Vec3 aiVector3DToVec3(const aiVector3D &aiVec) {
	return Vec3{ aiVec.x, aiVec.y, aiVec.z };
};

Vec3 aiVector3DToVec3(const aiColor3D &aiVec) {
	return Vec3{ aiVec.r, aiVec.g, aiVec.b };
};

void traverseAssimpScene(aiNode *node, const aiScene *aiScene, Node *parentNode, Scene &scene, SizeType &vertexOffset, SizeType &indexOffset) {
	dassert(aiScene != nullptr);

	if (node == nullptr || aiScene == nullptr) {
		return;
	}

	if (node == aiScene->mRootNode) {
		dassert(node->mNumMeshes != 0);

		for (unsigned int i = 0; i < aiScene->mNumLights; ++i) {
			aiLight *aiL = aiScene->mLights[i];
			if (aiL == nullptr) {
				continue;
			}

			LightNode *light = new LightNode;
			light->position = aiVector3DToVec3(aiL->mPosition);
			light->diffuse = aiVector3DToVec3(aiL->mColorDiffuse);
			light->specular = aiVector3DToVec3(aiL->mColorSpecular);
			light->ambient = aiVector3DToVec3(aiL->mColorAmbient);
			light->direction = aiVector3DToVec3(aiL->mDirection);
			light->attenuation = Vec3{aiL->mAttenuationConstant, aiL->mAttenuationLinear, aiL->mAttenuationQuadratic};
			light->innerAngleCutoff = aiL->mAngleInnerCone;
			light->outerAngleCutoff = aiL->mAngleOuterCone;
			
			scene.addNewLight(light);
		}

		for (unsigned int i = 0; i < aiScene->mNumCameras; ++i) {
			aiCamera *aiCam = aiScene->mCameras[i];
			if (aiCam == nullptr) {
				continue;
			}

			Camera cam;

			if (std::fabs(aiCam->mOrthographicWidth) < 1e-6f) {
				cam = Camera::perspectiveCamera(
					aiVector3DToVec3(aiCam->mPosition),
					aiCam->mHorizontalFOV,
					aiCam->mAspect,
					aiCam->mClipPlaneNear,
					aiCam->mClipPlaneFar
				);
			} else {
				cam = Camera::orthographicCamera(
					aiVector3DToVec3(aiCam->mPosition),
					2 * aiCam->mOrthographicWidth,
					2 * aiCam->mOrthographicWidth / aiCam->mAspect,
					aiCam->mClipPlaneNear,
					aiCam->mClipPlaneFar
				);
			}

			CameraNode *camNode = new CameraNode(std::move(cam));

			scene.addNewCamera(camNode);
		}
	}

	ModelNode *model = new ModelNode;
	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		aiMesh *mesh = aiScene->mMeshes[node->mMeshes[i]];
		
		// Setup the mesh
		Mesh resMesh;
		resMesh.indexOffset = indexOffset;
		resMesh.numIndices = mesh->mNumFaces ? mesh->mNumFaces * mesh->mFaces[0].mNumIndices : 0;
		resMesh.mat = readMaterialDataForMesh(mesh, aiScene, scene);

		// Make sure the next mesh knows where its indices begin
		indexOffset += resMesh.numIndices;

		// Read the mesh indices into the index buffer
		for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
			aiFace &face = mesh->mFaces[j];

			// We should have triangulated the mesh already
			dassert(face.mNumIndices == 3);
			
			for (unsigned int k = 0; k < face.mNumIndices; ++k) {
				// Adding the vertex offset here since these indices are zero-based
				// and the actual vertices of the current mesh begin at index vertexOffset
				// in the global scene.vertices array
				scene.indices.push_back(face.mIndices[k] + static_cast<unsigned int>(vertexOffset));
			}
		}
		model->meshes.push_back(resMesh);

		// save vertex data for the mesh in the global scene structure
		for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
			Vertex vertex;
			auto pos = mesh->mVertices[j];
			Vec3 p{ pos.x, pos.y, pos.z };

			auto normal = mesh->mNormals[j];
			Vec3 n{ normal.x, normal.y, normal.z };

			Vec2 uv{ 0.f, 0.f };
			if (mesh->HasTextureCoords(0)) {
				uv.x = mesh->mTextureCoords[0][j].x;
				uv.y = mesh->mTextureCoords[0][j].y;
			}

			vertex.pos = p;
			vertex.normal = n;
			vertex.uv = uv;

			scene.vertices.push_back(vertex);
		}
		vertexOffset += mesh->mNumVertices;
	}

	if (node->mNumMeshes > 0) {
		model->id = scene.nodes.size();
		scene.nodes.push_back(model);

		if (parentNode) {
			parentNode->children.push_back(model->id);
		}
	}

	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		traverseAssimpScene(node->mChildren[i], aiScene, model, scene, vertexOffset, indexOffset);
	}
}

// TODO: make own importer implementation. Should be able to import .obj, gltf2 files.
SceneLoaderError loadScene(const String &path, Scene &scene) {
	Assimp::Importer &importer = getAssimpImporter();

	const String ext = path.substr(path.find_last_of('.'));
	if (!importer.IsExtensionSupported(ext)) {
		return SceneLoaderError::UnsupportedExtention;
	}

	const aiScene *assimpScene = importer.ReadFile(
		path,
		aiProcess_Triangulate |
		aiProcess_OptimizeMeshes |
		aiProcess_JoinIdenticalVertices | 
		aiProcess_ConvertToLeftHanded
	);
	if (!assimpScene || assimpScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !assimpScene->mRootNode) {
		return SceneLoaderError::InvalidScene;
	}

	SizeType vertexOffset = 0;
	SizeType indexOffset = 0;
	traverseAssimpScene(assimpScene->mRootNode, assimpScene, nullptr, scene, vertexOffset, indexOffset);

	return SceneLoaderError::Success;
}
