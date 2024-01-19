#include "scene_loader.h"
#include "scene.h"

#include "framework/app.h"
#include "utils/defines.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "assimp/GltfMaterial.h"
#include "assimp/DefaultLogger.hpp"
#include "nlohmann/json.hpp"

#include <fstream>

struct JointData {
	SkeletonId skeletonId;
	uint8_t jointId;
};

struct AssimpSceneParseContext {
	const aiScene *aiScene = nullptr;
	aiNode *currentNode;
	Node *parentNode;
	Scene &scene;

	Vector<uint32_t> meshBaseVertex;
	Map<String, JointData> jointNameToData;

	JointData getJointData(const String &name) {
		if (auto it = jointNameToData.find(name); it != jointNameToData.end()) {
			return it->second;
		}

		dassert(false);
		return JointData{};
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

TextureId getTexture(aiMaterial *aiMat, aiTextureType aiType, Scene &scene) {
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

Mat4 aiMatrix4x4ToMat4(const aiMatrix4x4 &aiMat) {
	return Mat4{
		aiMat.a1, aiMat.a2, aiMat.a3, aiMat.a4,
		aiMat.b1, aiMat.b2, aiMat.b3, aiMat.b4,
		aiMat.c1, aiMat.c2, aiMat.c3, aiMat.c4,
		aiMat.d1, aiMat.d2, aiMat.d3, aiMat.d4
	};
}

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
	material.baseColorIndex = getTexture(aiMat, aiTextureType_BASE_COLOR, scene);
	material.normalsIndex = getTexture(aiMat, aiTextureType_NORMALS, scene);
	material.metallicRoughnessIndex = getTexture(aiMat, aiTextureType_METALNESS, scene);
	material.ambientOcclusionIndex = getTexture(aiMat, aiTextureType_AMBIENT_OCCLUSION, scene);

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

void readLights(const aiScene *aiScene, Scene &scene) {
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
		light->lightData.attenuation = Vec3{ aiL->mAttenuationConstant, aiL->mAttenuationLinear, aiL->mAttenuationQuadratic };
		light->lightData.innerAngleCutoff = aiL->mAngleInnerCone;
		light->lightData.outerAngleCutoff = aiL->mAngleOuterCone;

		scene.addNewLight(light);
	}
}

void readCameras(const aiScene *aiScene, Scene &scene) {
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

void readMeshBones(unsigned int meshId, AssimpSceneParseContext &ctx) {
	const aiScene *aiScene = ctx.aiScene;
	aiMesh *mesh = aiScene->mMeshes[meshId];

	for (unsigned int i = 0; i < mesh->mNumBones; ++i) {
		aiBone *bone = mesh->mBones[i];
		auto jointData = ctx.getJointData(String{ bone->mName.C_Str(), bone->mName.length });

		ctx.scene.animationManager.skeletons[jointData.skeletonId].joints[jointData.jointId].invBindPose = aiMatrix4x4ToMat4(bone->mOffsetMatrix);

		for (unsigned int j = 0; j < bone->mNumWeights; ++j) {
			const auto &vertexWeight = bone->mWeights[j];
			const auto vertexId = ctx.meshBaseVertex[meshId] + vertexWeight.mVertexId;
			ctx.scene.animatedData.vertexBuffer[vertexId].addJointData(jointData.jointId, vertexWeight.mWeight);

			LOG_FMT(Info, "Bone %d affects vertex %d by %f", jointData.jointId, vertexId, vertexWeight.mWeight);
		}
	}
}

void readJoint(AssimpSceneParseContext &ctx, Skeleton &skeleton, uint8_t parent) {
	Joint joint = {};

	aiNode *jointNode = ctx.currentNode;
	joint.name = String{ jointNode->mName.C_Str(), jointNode->mName.length };
	joint.parent = parent;

	dassert(skeleton.joints.size() < 256);
	uint8_t jointId = static_cast<uint8_t>(skeleton.joints.size());
	ctx.jointNameToData[joint.name] = JointData{ .skeletonId = skeleton.id, .jointId = jointId };
	skeleton.joints.push_back(joint);

	for (unsigned int i = 0; i < jointNode->mNumChildren; ++i) {
		ctx.currentNode = jointNode->mChildren[i];
		readJoint(ctx, skeleton, jointId);
	}
}

SkeletonId readSkeleton(AssimpSceneParseContext &ctx) {
	aiNode *jointNode = ctx.currentNode;
	if (auto it = ctx.jointNameToData.find(String{ jointNode->mName.C_Str(), jointNode->mName.length}); it != ctx.jointNameToData.end()) {
		return it->second.skeletonId;
	}

	Skeleton skeleton = {};
	skeleton.id = ctx.scene.animationManager.skeletons.size();

	readJoint(ctx, skeleton, -1);

	ctx.scene.animationManager.skeletons.push_back(skeleton);
	return ctx.scene.animationManager.skeletons.size() - 1;
}

void readAnimations(AssimpSceneParseContext &ctx) {
	const aiScene *aiScene = ctx.aiScene;

	for (unsigned int i = 0; i < aiScene->mNumAnimations; ++i) {
		auto anim = aiScene->mAnimations[i];
		
		AnimationClip clip = {};
		const auto animationClipName = String{ anim->mName.C_Str(), anim->mName.length };

		// TODO: ...

		ctx.scene.animationManager.nameToAnimation[animationClipName] = clip;
	}
}

void readStaticMeshes(AssimpSceneParseContext &ctx) {
	aiNode *node = ctx.currentNode;
	const aiScene *aiScene = ctx.aiScene;
	Scene &scene = ctx.scene;

	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		aiMesh *mesh = aiScene->mMeshes[node->mMeshes[i]];

		auto vertexOffset = scene.staticData.vertexCount();
		auto indexOffset = scene.staticData.indexCount();

		// Setup the mesh
		Mesh resMesh;
		resMesh.indexOffset = indexOffset;
		resMesh.numIndices = mesh->mNumFaces ? mesh->mNumFaces * mesh->mFaces[0].mNumIndices : 0;
		resMesh.mat = readMaterialDataForMesh(mesh, aiScene, scene);

		// save vertex data for the mesh in the global scene structure
		for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
			StaticVertex vertex;
			vertex.pos = aiVector3DToVec3(mesh->mVertices[j]);

			if (mesh->HasTextureCoords(0)) {
				vertex.uv.x = mesh->mTextureCoords[0][j].x;
				vertex.uv.y = mesh->mTextureCoords[0][j].y;
			}

			if (mesh->HasNormals()) {
				vertex.normal = aiVector3DToVec3(mesh->mNormals[j]);
			}

			if (mesh->HasTangentsAndBitangents()) {
				vertex.tangent = aiVector3DToVec3(mesh->mTangents[j]);
			}
			
			scene.staticData.vertexBuffer.push_back(static_cast<StaticVertex>(vertex));
			scene.sceneBox.addPoint(vertex.pos);
		}

		// Read the mesh indices into the index buffer
		for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
			aiFace &face = mesh->mFaces[j];

			// We should have triangulated the mesh already
			dassert(face.mNumIndices == 3);

			Vector<unsigned int> &indices = scene.staticData.indexBuffer;

			for (unsigned int k = 0; k < face.mNumIndices; ++k) {
				// Adding the vertex offset here since these indices are zero-based
				// and the actual vertices of the current mesh begin at index vertexOffset
				// in the global scene.vertices array
				unsigned int offsetIndex = face.mIndices[k] + static_cast<unsigned int>(vertexOffset);

				indices.push_back(offsetIndex);
			}

			// Generate normals and tangents if the mesh doesn't contain them
			const SizeType index = indices.size() - 3;
			if (!mesh->HasNormals() || !mesh->HasTangentsAndBitangents()) {
				StaticVertex *v[3];
				v[0] = &scene.staticData.vertexBuffer[indices[index + 0]];
				v[1] = &scene.staticData.vertexBuffer[indices[index + 1]];
				v[2] = &scene.staticData.vertexBuffer[indices[index + 2]];

				Vec3 edge0 = v[1]->pos - v[0]->pos;
				Vec3 edge1 = v[2]->pos - v[0]->pos;

				if (!mesh->HasNormals()) {
					v[0]->normal = v[1]->normal = v[2]->normal = glm::normalize(glm::cross(edge0, edge1));
				}

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
			}
		}

		scene.staticData.meshes.push_back(resMesh);
	}
}

void readSkinnedMeshes(AssimpSceneParseContext &ctx) {
	aiNode *node = ctx.currentNode;
	const aiScene *aiScene = ctx.aiScene;
	Scene &scene = ctx.scene;

	ctx.meshBaseVertex.resize(aiScene->mNumMeshes);
	for (unsigned int i = 0; i < aiScene->mNumMeshes; ++i) {
		aiMesh *mesh = aiScene->mMeshes[i];

		dassert(mesh->HasBones());
		auto vertexOffset = scene.animatedData.vertexCount();
		auto indexOffset = scene.animatedData.indexCount();
		ctx.meshBaseVertex[i] = vertexOffset;

		// Setup the mesh
		Mesh resMesh;
		resMesh.indexOffset = indexOffset;
		resMesh.numIndices = mesh->mNumFaces ? mesh->mNumFaces * mesh->mFaces[0].mNumIndices : 0;
		resMesh.mat = readMaterialDataForMesh(mesh, aiScene, scene);

		// save vertex data for the mesh in the global scene structure
		for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
			AnimatedVertex vertex;
			vertex.pos = aiVector3DToVec3(mesh->mVertices[j]);

			if (mesh->HasTextureCoords(0)) {
				vertex.uv.x = mesh->mTextureCoords[0][j].x;
				vertex.uv.y = mesh->mTextureCoords[0][j].y;
			}

			if (mesh->HasNormals()) {
				vertex.normal = aiVector3DToVec3(mesh->mNormals[j]);
			}

			if (mesh->HasTangentsAndBitangents()) {
				vertex.tangent = aiVector3DToVec3(mesh->mTangents[j]);
			}

			scene.animatedData.vertexBuffer.push_back(vertex);
		}

		// Read the mesh indices into the index buffer
		for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
			aiFace &face = mesh->mFaces[j];

			// We should have triangulated the mesh already
			dassert(face.mNumIndices == 3);

			Vector<unsigned int> &indices = scene.animatedData.indexBuffer;

			for (unsigned int k = 0; k < face.mNumIndices; ++k) {
				// Adding the vertex offset here since these indices are zero-based
				// and the actual vertices of the current mesh begin at index vertexOffset
				// in the global scene.vertices array
				unsigned int offsetIndex = face.mIndices[k] + static_cast<unsigned int>(vertexOffset);
				
				indices.push_back(offsetIndex);
			}

			// Generate normals and tangents if the mesh doesn't contain them
			const SizeType index = indices.size() - 3;
			if (!mesh->HasNormals() || !mesh->HasTangentsAndBitangents()) {
				AnimatedVertex *v[3];
				v[0] = &scene.animatedData.vertexBuffer[indices[index + 0]];
				v[1] = &scene.animatedData.vertexBuffer[indices[index + 1]];
				v[2] = &scene.animatedData.vertexBuffer[indices[index + 2]];

				Vec3 edge0 = v[1]->pos - v[0]->pos;
				Vec3 edge1 = v[2]->pos - v[0]->pos;

				if (!mesh->HasNormals()) {
					v[0]->normal = v[1]->normal = v[2]->normal = glm::normalize(glm::cross(edge0, edge1));
				}

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
			}
		}

		readMeshBones(i, ctx);
		scene.animatedData.meshes.push_back(resMesh);

		// Add the number of vertices we added to the global scene vertex buffer
		// so the next mesh's indices are offset correctly.
		vertexOffset += mesh->mNumVertices;
	}
}

void traverseAssimpScene(AssimpSceneParseContext &ctx) {
	aiNode *node = ctx.currentNode;
	const aiScene *aiScene = ctx.aiScene;
	Scene &scene = ctx.scene;
	Node *parentNode = ctx.parentNode;

	dassert(aiScene != nullptr);

	if (node == nullptr || aiScene == nullptr) {
		return;
	}

	if (node == aiScene->mRootNode) {
		readLights(aiScene, scene);

		readCameras(aiScene, scene);

		// If the root node of the scene is a Skin we only have 1 skinned mesh
		if (node->mName == aiString{ "Skin" }) {
			dassert(aiScene->HasAnimations());

			uint32_t rootJointChildId = -1;
			for (int i = 0; i < node->mNumChildren; ++i) {
				if (node->mChildren[i]->mNumMeshes == 0) {
					dassert(rootJointChildId == -1);
					rootJointChildId = i;
				}
			}

			ctx.currentNode = node->mChildren[rootJointChildId];
			SkeletonId skeletonId = readSkeleton(ctx);
			ctx.currentNode = node;

			readAnimations(ctx);

			auto meshOffset = ctx.scene.animatedData.meshes.size();
			readSkinnedMeshes(ctx);

			AnimatedModelNode *modelNode = new AnimatedModelNode;
			modelNode->startMesh = meshOffset;
			modelNode->numMeshes = ctx.scene.animatedData.meshes.size() - meshOffset;
			modelNode->currentAnimation = INVALID_ANIMATION_ID;
			modelNode->skeletonId = skeletonId;
			modelNode->id = ctx.scene.nodes.size();
			ctx.scene.nodes.push_back(modelNode);

			return;
		}
	}

	Node *currentNode = nullptr;
	if (node->mNumMeshes > 0) {
		ModelNode *model = new ModelNode;
		model->startMesh = scene.staticData.meshes.size();
		model->numMeshes = node->mNumMeshes;
		model->id = scene.nodes.size();
		scene.nodes.push_back(model);

		readStaticMeshes(ctx);

		if (parentNode) {
			parentNode->children.push_back(model->id);
		}
		currentNode = model;
	} else {
		LOG_FMT(Info, "Assimp node (%s) has no meshes", node->mName);
	}

	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		ctx.currentNode = node->mChildren[i];
		ctx.parentNode = currentNode;
		traverseAssimpScene(ctx);
	}
}

struct AssimpLogger {
	AssimpLogger() {
		Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);
	}

	~AssimpLogger() {
		Assimp::DefaultLogger::kill();
	}
};

SceneLoaderError loadAssimpScene(const std::filesystem::path &path, Scene &scene) {
	LOG(Info, "SceneLoader::loadScene");

	Assimp::Importer &importer = getAssimpImporter();

	const String ext = path.extension().string();
	if (!importer.IsExtensionSupported(ext)) {
		return SceneLoaderError::UnsupportedExtention;
	}

	AssimpLogger logger;

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

	AssimpSceneParseContext ctx {
		.aiScene = assimpScene,
		.currentNode = assimpScene->mRootNode,
		.parentNode = nullptr,
		.scene = scene,
	};
	traverseAssimpScene(ctx);

	LOG(Info, "SceneLoader::loadScene SUCCESS");
	return SceneLoaderError::Success;
}

Vec3 vec3FromJson(const nlohmann::json &data) {
	return Vec3{ data[0].get<float>(), data[1].get<float>(), data[2].get<float>() };
}

Vec4 vec4FromJson(const nlohmann::json &data) {
	return Vec4{ data[0].get<float>(), data[1].get<float>(), data[2].get<float>(), data[3].get<float>() };
}

Mat4 transformFromJson(const nlohmann::json &data) {
	Mat4 trans{ vec4FromJson(data), vec4FromJson(data + 4), vec4FromJson(data + 8), Vec4(0, 0, 0, 1) };
	return trans;
}

SceneLoaderError loadScene(const String &path, Scene &outScene) {
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
	auto res = loadAssimpScene(rootDir / staticPath.get<String>(), outScene);
	if (res != SceneLoaderError::Success) {
		return res;
	}

	json animatedPath = scene["animated"];
	res = loadAssimpScene(rootDir / animatedPath.get<String>(), outScene);
	if (res != SceneLoaderError::Success) {
		return res;
	}

	auto lights = scene["lights"];
	for (auto &l : lights) {
		json light = l["light"];
		String lightType = light["type"];
		Vec3 diffuse = vec3FromJson(light["diffuse"]);
		Vec3 specular = vec3FromJson(light["specular"]);
		Vec3 ambient = vec3FromJson(light["ambient"]);

		if (lightType == "directional") {
			LightNode *lDir = new LightNode(
				LightNode::directional(
					vec3FromJson(light["direction"]),
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
			lPoint->lightData.position = vec3FromJson(light["position"]);
			lPoint->lightData.diffuse = diffuse;
			lPoint->lightData.ambient = ambient;
			lPoint->lightData.specular = specular;
			lPoint->lightData.attenuation = vec3FromJson(light["attenuation"]);
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
				vec3FromJson(camera["position"]),
				camera["fov"].get<float>(),
				app->getWidth() / static_cast<float>(app->getHeight()),
				camera["nearPlane"].get<float>(),
				camera["farPlane"].get<float>()
			);
			cam.setKeepXZPlane(true);
		}

		if (cameraType == "orthographic") {
			cam = Dar::Camera::orthographicCamera(
				vec3FromJson(camera["position"]),
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