#include "scene_loader.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "assimp/GltfMaterial.h"

#include "MikkTSpace/mikktspace.h"

#include "d3d12_defines.h"

#define USE_MIKKTSPACE

struct MeshData {
	Scene *scene;
	aiMesh *mesh;
	Map<UINT, UINT> *indicesMapping;
};

/// Helper structure for generating the tangents with the MikkTSpace algorithm
/// as given by glTF2.0 specs.
struct MikkTSpaceTangentSpaceGenerator {
	void init(MeshData *meshData) {
		tSpaceIface.m_getNumFaces = getNumFaces;
		tSpaceIface.m_getNormal = getNormal;
		tSpaceIface.m_getNumVerticesOfFace = getNumVerticesOfFace;
		tSpaceIface.m_getPosition = getPosition;
		tSpaceIface.m_getTexCoord = getTexCoord;
		tSpaceIface.m_setTSpaceBasic = setTSpaceBasic;

		tSpaceCtx.m_pInterface = &tSpaceIface;
		tSpaceCtx.m_pUserData = reinterpret_cast<void*>(meshData);
	}

	tbool generateTangets() {
		return genTangSpaceDefault(&tSpaceCtx);
	}

private:
	SMikkTSpaceInterface tSpaceIface = {};
	SMikkTSpaceContext tSpaceCtx = {};

	static int getNumFaces(const SMikkTSpaceContext *pContext) {
		MeshData *meshData = static_cast<MeshData*>(pContext->m_pUserData);
		return meshData->mesh->mNumFaces;
	}

	static int getNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
		return 3; // We always triangulate the imported mesh
	}

	static void getPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
		MeshData *meshData = static_cast<MeshData *>(pContext->m_pUserData);
		aiMesh *mesh = meshData->mesh;
		aiVector3D &v = mesh->mVertices[mesh->mFaces[iFace].mIndices[iVert]];
		fvPosOut[0] = v.x;
		fvPosOut[1] = v.y;
		fvPosOut[2] = v.z;
	}

	static void getNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
		MeshData *meshData = static_cast<MeshData *>(pContext->m_pUserData);
		aiMesh *mesh = meshData->mesh;
		aiVector3D &v = mesh->mNormals[mesh->mFaces[iFace].mIndices[iVert]];
		fvNormOut[0] = v.x;
		fvNormOut[1] = v.y;
		fvNormOut[2] = v.z;
	}

	static void getTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
		MeshData *meshData = static_cast<MeshData *>(pContext->m_pUserData);
		aiMesh *mesh = meshData->mesh;
		aiVector3D &uv = mesh->mTextureCoords[0][mesh->mFaces[iFace].mIndices[iVert]];
		fvTexcOut[0] = uv.x;
		fvTexcOut[1] = uv.y;
	}

	static void setTSpaceBasic(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
		MeshData *meshData = static_cast<MeshData*>(pContext->m_pUserData);
		Scene &scene = *meshData->scene;
		const aiMesh &mesh = *meshData->mesh;
		const auto &indicesMap = *meshData->indicesMapping;
		const UINT offsetIndex = indicesMap.at(mesh.mFaces[iFace].mIndices[iVert]);
		
		Vertex &v = scene.vertices[offsetIndex];
		v.tangent.x = fvTangent[0];
		v.tangent.y = fvTangent[1];
		v.tangent.z = fvTangent[2];

		// TODO: Ignore the sign for now. See if that's needed.
		// v.tangentSign = fSign;
	}
};

// !!! Note: not thread-safe. Generally an inctance of the Assimp::Importer should be used by one thread only!
static Assimp::Importer *importer = nullptr;
Assimp::Importer& getAssimpImporter() {
	if (importer == nullptr) {
		importer = new Assimp::Importer;
	}

	return *importer;
}

TextureId loadTexture(aiMaterial *aiMat, aiTextureType aiType, TextureType matType, Scene &scene) {
	const int matTypeCount = aiMat->GetTextureCount(aiType);
	if (matTypeCount <= 0 && aiType != aiTextureType_METALNESS) {
		return INVALID_TEXTURE_ID;
	}

	// Only read the first texture, for now we won't export
	// multiple textures per channel.
	aiString path;

	// Assimp + glTF pain
	if (aiType == aiTextureType_METALNESS) {
		aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &path);
	} else {
		aiMat->GetTexture(aiType, 0, &path);
	}

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

Vec3 aiVector3DToVec3(const aiVector3D &aiVec) {
	return Vec3{ aiVec.x, aiVec.y, aiVec.z };
};

Vec3 aiVector3DToVec3(const aiColor3D &aiVec) {
	return Vec3{ aiVec.r, aiVec.g, aiVec.b };
};

MaterialId readMaterialDataForMesh(aiMesh *mesh, const aiScene *sc, Scene &scene) {
	auto matIdx = mesh->mMaterialIndex;
	if (matIdx < 0) {
		return INVALID_MATERIAL_ID;
	}

	aiMaterial *aiMat = sc->mMaterials[matIdx];

	// Blinn-Phong model materials
	/*
	TextureId diffuse = loadTexture(aiMat, aiTextureType_DIFFUSE, TextureType::Diffuse, scene);
	TextureId specular = loadTexture(aiMat, aiTextureType_SPECULAR, TextureType::Specular, scene);
	TextureId normals = loadTexture(aiMat, aiTextureType_NORMALS, TextureType::Normals, scene);
	*/

	// PBR model materials
	MaterialData material;
	material.baseColor = loadTexture(aiMat, aiTextureType_BASE_COLOR, TextureType::BaseColor, scene);
	material.normals   = loadTexture(aiMat, aiTextureType_NORMALS, TextureType::BaseColor, scene);
	material.metallicRoughness = loadTexture(aiMat, aiTextureType_METALNESS, TextureType::BaseColor, scene);
	material.ambientOcclusion = loadTexture(aiMat, aiTextureType_AMBIENT_OCCLUSION, TextureType::BaseColor, scene);

	ai_real metallicFactor, roughnessFactor;
	aiVector3D baseColorFactor;
	aiReturn result = aiMat->Get(AI_MATKEY_BASE_COLOR, baseColorFactor);
	material.baseColorFactor = result == AI_SUCCESS ? aiVector3DToVec3(baseColorFactor) : Vec3(1.f);

	result = aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor);
	material.metallicFactor = result == AI_SUCCESS ? metallicFactor : 1.f;
	
	result = aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor);
	material.roughnessFactor = result == AI_SUCCESS ? roughnessFactor : 1.f;

	return scene.getNewMaterial(material);
}

void traverseAssimpScene(aiNode *node, const aiScene *aiScene, Node *parentNode, Scene &scene, SizeType &vertexOffset, SizeType &indexOffset, SceneLoaderFlags flags) {
	dassert(aiScene != nullptr);

	if (node == nullptr || aiScene == nullptr) {
		return;
	}

	if (node == aiScene->mRootNode) {
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

		const bool genTangents = mesh->HasTextureCoords(0) && ((flags & sceneLoaderFlags_overrideGenTangents) || !mesh->HasTangentsAndBitangents());

		// Setup the mesh
		Mesh resMesh;
		resMesh.indexOffset = indexOffset;
		resMesh.numIndices = mesh->mNumFaces ? mesh->mNumFaces * mesh->mFaces[0].mNumIndices : 0;
		resMesh.mat = readMaterialDataForMesh(mesh, aiScene, scene);

		// Make sure the next mesh knows where its indices begin
		indexOffset += resMesh.numIndices;

		// save vertex data for the mesh in the global scene structure
		for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
			Vertex vertex;
			vertex.pos = aiVector3DToVec3(mesh->mVertices[j]);

			if (mesh->HasTextureCoords(0)) {
				vertex.uv.x = mesh->mTextureCoords[0][j].x;
				vertex.uv.y = mesh->mTextureCoords[0][j].y;
			}

			if (mesh->HasNormals()) {
				vertex.normal = aiVector3DToVec3(mesh->mNormals[j]);
			}

			if (mesh->HasTangentsAndBitangents() && !genTangents) {
				vertex.tangent = aiVector3DToVec3(mesh->mTangents[j]);
			}

			scene.vertices.push_back(vertex);
		}

		MikkTSpaceTangentSpaceGenerator tangentGenerator = {};
		MeshData meshData = {};
		Map<UINT, UINT> indicesMapping;
		if (genTangents) {
			meshData.scene = &scene;
			meshData.mesh = mesh;
			meshData.indicesMapping = &indicesMapping;
			tangentGenerator.init(&meshData);
		}

		// Read the mesh indices into the index buffer
		for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
			aiFace &face = mesh->mFaces[j];

			// We should have triangulated the mesh already
			dassert(face.mNumIndices == 3);
			
			for (unsigned int k = 0; k < face.mNumIndices; ++k) {
				// Adding the vertex offset here since these indices are zero-based
				// and the actual vertices of the current mesh begin at index vertexOffset
				// in the global scene.vertices array
				unsigned int offsetIndex = face.mIndices[k] + static_cast<unsigned int>(vertexOffset);
				scene.indices.push_back(offsetIndex);

				if (genTangents) {
					indicesMapping[face.mIndices[k]] = offsetIndex;
				}
			}

			// Generate normals and tangents if the mesh doesn't contain them
			const int index = scene.indices.size() - 3;
			if (!mesh->HasNormals() || !mesh->HasTangentsAndBitangents() || (genTangents && !mesh->HasNormals())) {
				Vertex *v[3];
				v[0] = &scene.vertices[scene.indices[index + 0]];
				v[1] = &scene.vertices[scene.indices[index + 1]];
				v[2] = &scene.vertices[scene.indices[index + 2]];

				Vec3 edge0 = v[1]->pos - v[0]->pos;
				Vec3 edge1 = v[2]->pos - v[0]->pos;

				if (!mesh->HasNormals()) {
					v[0]->normal = v[1]->normal = v[2]->normal = edge0.cross(edge1).normalized();
				}

#ifndef USE_MIKKTSPACE
				// Check for texture coordinates before generating the tangent vector
				if (!mesh->HasTangentsAndBitangents() && mesh->HasTextureCoords(0)) {
					Vec2 &uv0 = v[0]->uv;
					Vec2 &uv1 = v[1]->uv;
					Vec2 &uv2 = v[2]->uv;

					Vec2 dUV0 = uv1 - uv0;
					Vec2 dUV1 = uv2 - uv0;

					v[0]->tangent.x = v[1]->tangent.x = v[2]->tangent.x = dUV1.y * edge0.x - dUV0.y * edge1.x;
					v[0]->tangent.y = v[1]->tangent.y = v[2]->tangent.y = dUV1.y * edge0.y - dUV0.y * edge1.y;
					v[0]->tangent.z = v[1]->tangent.z = v[2]->tangent.z = dUV1.y * edge0.z - dUV0.y * edge1.z;
				}
#endif // !USE_MIKKTSPACE
			}
		}
		model->meshes.push_back(resMesh);

#ifdef USE_MIKKTSPACE
		if (genTangents) {
			tbool result = tangentGenerator.generateTangets();
			dassert(result);
		}
#endif // USE_MIKKTSPACE

		// Add the number of vertices we added to the global scene vertex buffer
		// so the next mesh's indices are offset correctly.
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
		traverseAssimpScene(node->mChildren[i], aiScene, model, scene, vertexOffset, indexOffset, flags);
	}
}

// TODO: make own importer implementation. Should be able to import .obj, gltf2 files.
SceneLoaderError loadScene(const String &path, Scene &scene, SceneLoaderFlags flags) {
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
		aiProcess_CalcTangentSpace |
		aiProcess_ConvertToLeftHanded
	);
	if (!assimpScene || assimpScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !assimpScene->mRootNode) {
		return SceneLoaderError::InvalidScene;
	}

	SizeType vertexOffset = 0;
	SizeType indexOffset = 0;
	traverseAssimpScene(assimpScene->mRootNode, assimpScene, nullptr, scene, vertexOffset, indexOffset, flags);

	return SceneLoaderError::Success;
}
