#include "scene_loader.h"
#include "scene.h"

#include "framework/app.h"
#include "utils/defines.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "assimp/GltfMaterial.h"
#include "MikkTSpace/mikktspace.h"
#include "nlohmann/json.hpp"

#include <fstream>

#define USE_MIKKTSPACE

struct MikkTSpaceMeshData {
	Scene *scene;
	aiMesh *mesh;
	Map<UINT, UINT> *indicesMapping;
};

/// Helper structure for generating the tangents with the MikkTSpace algorithm
/// as given by glTF2.0 specs.
struct MikkTSpaceTangentSpaceGenerator {
	void init(MikkTSpaceMeshData *meshData) {
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
		MikkTSpaceMeshData *meshData = static_cast<MikkTSpaceMeshData *>(pContext->m_pUserData);
		return meshData->mesh->mNumFaces;
	}

	static int getNumVerticesOfFace(const SMikkTSpaceContext */*pContext*/, const int /*iFace*/) {
		return 3; // We always triangulate the imported mesh
	}

	static void getPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert) {
		MikkTSpaceMeshData *meshData = static_cast<MikkTSpaceMeshData*>(pContext->m_pUserData);
		aiMesh *mesh = meshData->mesh;
		aiVector3D &v = mesh->mVertices[mesh->mFaces[iFace].mIndices[iVert]];
		fvPosOut[0] = v.x;
		fvPosOut[1] = v.y;
		fvPosOut[2] = v.z;
	}

	static void getNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert) {
		MikkTSpaceMeshData *meshData = static_cast<MikkTSpaceMeshData*>(pContext->m_pUserData);
		aiMesh *mesh = meshData->mesh;
		aiVector3D &v = mesh->mNormals[mesh->mFaces[iFace].mIndices[iVert]];
		fvNormOut[0] = v.x;
		fvNormOut[1] = v.y;
		fvNormOut[2] = v.z;
	}

	static void getTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert) {
		MikkTSpaceMeshData *meshData = static_cast<MikkTSpaceMeshData*>(pContext->m_pUserData);
		aiMesh *mesh = meshData->mesh;
		aiVector3D &uv = mesh->mTextureCoords[0][mesh->mFaces[iFace].mIndices[iVert]];
		fvTexcOut[0] = uv.x;
		fvTexcOut[1] = uv.y;
	}

	static void setTSpaceBasic(const SMikkTSpaceContext *pContext, const float fvTangent[], const float /*fSign*/, const int iFace, const int iVert) {
		MikkTSpaceMeshData *meshData = static_cast<MikkTSpaceMeshData*>(pContext->m_pUserData);
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

// !!! Note: not thread-safe. Generally an instance of the Assimp::Importer should be used by one thread only!
static Assimp::Importer *importer_ = nullptr;
Assimp::Importer& getAssimpImporter() {
	if (importer_ == nullptr) {
		importer_ = new Assimp::Importer;
	}

	return *importer_;
}

TextureId loadTexture(aiMaterial *aiMat, aiTextureType aiType, Scene &scene) {
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

	return scene.getNewTexture(path.C_Str());
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
	material.baseColorIndex = loadTexture(aiMat, aiTextureType_BASE_COLOR, scene);
	material.normalsIndex = loadTexture(aiMat, aiTextureType_NORMALS, scene);
	material.metallicRoughnessIndex = loadTexture(aiMat, aiTextureType_METALNESS, scene);
	material.ambientOcclusionIndex = loadTexture(aiMat, aiTextureType_AMBIENT_OCCLUSION, scene);

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
			light->lightData.position = aiVector3DToVec3(aiL->mPosition);
			light->lightData.diffuse = aiVector3DToVec3(aiL->mColorDiffuse);
			light->lightData.specular = aiVector3DToVec3(aiL->mColorSpecular);
			light->lightData.ambient = aiVector3DToVec3(aiL->mColorAmbient);
			light->lightData.direction = aiVector3DToVec3(aiL->mDirection);
			light->lightData.attenuation = Vec3{aiL->mAttenuationConstant, aiL->mAttenuationLinear, aiL->mAttenuationQuadratic};
			light->lightData.innerAngleCutoff = aiL->mAngleInnerCone;
			light->lightData.outerAngleCutoff = aiL->mAngleOuterCone;
			
			scene.addNewLight(light);
		}

		for (unsigned int i = 0; i < aiScene->mNumCameras; ++i) {
			aiCamera *aiCam = aiScene->mCameras[i];
			if (aiCam == nullptr) {
				continue;
			}

			Dar::Camera cam;

			if (std::fabs(aiCam->mOrthographicWidth) < 1e-6f) {
				cam = Dar::Camera::perspectiveCamera(
					aiVector3DToVec3(aiCam->mPosition),
					aiCam->mHorizontalFOV,
					aiCam->mAspect,
					aiCam->mClipPlaneNear,
					aiCam->mClipPlaneFar
				);
			} else {
				cam = Dar::Camera::orthographicCamera(
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
	if (node->mNumMeshes > 0) {
		model->startMesh = scene.meshes.size();
		model->numMeshes = node->mNumMeshes;
	}
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
			scene.sceneBox.addPoint(vertex.pos);
		}

		MikkTSpaceTangentSpaceGenerator tangentGenerator = {};
		MikkTSpaceMeshData meshData = {};
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
			const SizeType index = scene.indices.size() - 3;
			if (!mesh->HasNormals() || !mesh->HasTangentsAndBitangents() || (genTangents && !mesh->HasNormals())) {
				Vertex *v[3];
				v[0] = &scene.vertices[scene.indices[index + 0]];
				v[1] = &scene.vertices[scene.indices[index + 1]];
				v[2] = &scene.vertices[scene.indices[index + 2]];

				Vec3 edge0 = v[1]->pos - v[0]->pos;
				Vec3 edge1 = v[2]->pos - v[0]->pos;

				if (!mesh->HasNormals()) {
					v[0]->normal = v[1]->normal = v[2]->normal = glm::normalize(glm::cross(edge0, edge1));
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
		scene.meshes.push_back(resMesh);

#ifdef USE_MIKKTSPACE
		if (genTangents) {
#pragma warning(suppress: 4189)
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
SceneLoaderError loadStatic(const std::filesystem::path &path, Scene &scene, SceneLoaderFlags flags) {
	LOG(Info, "SceneLoader::loadScene");

	Assimp::Importer &importer = getAssimpImporter();

	const String ext = path.extension().string();
	if (!importer.IsExtensionSupported(ext)) {
		return SceneLoaderError::UnsupportedExtention;
	}

	const aiScene *assimpScene = importer.ReadFile(
		path.string(),
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

	LOG(Info, "SceneLoader::loadScene SUCCESS");
	return SceneLoaderError::Success;
}

Vec3 fromJson(const nlohmann::json &data) {
	return Vec3{ data[0].get<float>(), data[1].get<float>(), data[2].get<float>() };
}

SceneLoaderError loadScene(const String &path, Scene &outScene, SceneLoaderFlags flags) {
	using json = nlohmann::json;

	auto p = std::filesystem::path(path);
	if (!std::filesystem::exists(p)) {
		LOG_FMT(Error, "Scene file %s does not exist!", path.c_str());
		return SceneLoaderError::InvalidScenePath;
	}

	auto rootDir = p.parent_path();

	std::ifstream f(path);
	if (!f.is_open()) {
		LOG_FMT(Error, "Failed to open %s. Error: %s", path.c_str(), strerror(errno));
		return SceneLoaderError::CorruptSceneFile;
	}
	json data = json::parse(f);

	json scene = data["scene"];
	json staticPath = scene["static"];
	auto res = loadStatic(rootDir / staticPath.get<String>(), outScene, flags);
	if (res != SceneLoaderError::Success) {
		return res;
	}

	auto lights = scene["lights"];
	for (auto &l : lights) {
		json light = l["light"];
		String lightType = light["type"];
		Vec3 diffuse = fromJson(light["diffuse"]);
		Vec3 specular = fromJson(light["specular"]);
		Vec3 ambient = fromJson(light["ambient"]);

		if (lightType == "directional") {
			LightNode *lDir = new LightNode(
				LightNode::directional(
					fromJson(light["direction"]),
					diffuse,
					specular,
					ambient
				)
			);
			outScene.addNewLight(lDir);
		}

		if (lightType == "spot") {
			LightNode *lSpot = new LightNode(
				LightNode::spot(
					diffuse,
					ambient,
					specular,
					glm::radians(light["innerCutoff"].get<float>()),
					glm::radians(light["outerCutoff"].get<float>())
				)
			);
			outScene.addNewLight(lSpot);
		}

		if (lightType == "point") {
			LightNode *lPoint = new LightNode;
			lPoint->lightData.type = LightType::Point;
			lPoint->lightData.position = fromJson(light["position"]);
			lPoint->lightData.diffuse = diffuse;
			lPoint->lightData.ambient = ambient;
			lPoint->lightData.specular = specular;
			lPoint->lightData.attenuation = fromJson(light["attenuation"]);
			outScene.addNewLight(lPoint);
		}
	}

	auto app = Dar::getApp();
	{
		json camera = scene["camera"];
		const String cameraType = camera["type"];
		Dar::Camera cam;
		if (cameraType == "perspective") {
			cam = Dar::Camera::perspectiveCamera(
				fromJson(camera["position"]), 
				camera["fov"].get<float>(),
				app->getWidth() / static_cast<float>(app->getHeight()),
				camera["nearPlane"].get<float>(),
				camera["farPlane"].get<float>()
			);
			cam.setKeepXZPlane(true);
		}

		if (cameraType == "orthographic") {
			cam = Dar::Camera::orthographicCamera(
				fromJson(camera["position"]),
				camera["rectWidth"].get<float>(),
				camera["rectHeight"].get<float>(),
				camera["nearPlane"].get<float>(),
				camera["farPlane"].get<float>()
			);
		}
		CameraNode *camNode = new CameraNode(std::move(cam));
		outScene.addNewCamera(camNode);
	}

	return SceneLoaderError::Success;
}