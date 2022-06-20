#include "scene_loader.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "assimp/GltfMaterial.h"

#include "MikkTSpace/mikktspace.h"

#include "utils/defines.h"

#include "scene.h"

#define USE_MIKKTSPACE

struct MikkTSpaceMeshData {
	Scene *scene;
	aiMesh *mesh;
	Map<UINT, UINT> *indicesMapping;
	MeshType meshType;
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

	static int getNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace) {
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

	static void setTSpaceBasic(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
		MikkTSpaceMeshData *meshData = static_cast<MikkTSpaceMeshData*>(pContext->m_pUserData);
		Scene &scene = *meshData->scene;
		const aiMesh &mesh = *meshData->mesh;
		const auto &indicesMap = *meshData->indicesMapping;
		const UINT offsetIndex = indicesMap.at(mesh.mFaces[iFace].mIndices[iVert]);
		
		if (meshData->meshType == MeshType::Static) {
			auto &sv = scene.staticData.vertices[offsetIndex];
			sv.tangent.x = fvTangent[0];
			sv.tangent.y = fvTangent[1];
			sv.tangent.z = fvTangent[2];
		} else if (meshData->meshType == MeshType::Skinned) {
			auto &mv = scene.movingData.vertices[offsetIndex];
			mv.tangent.x = fvTangent[0];
			mv.tangent.y = fvTangent[1];
			mv.tangent.z = fvTangent[2];
		}

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
	material.baseColorIndex = loadTexture(aiMat, aiTextureType_BASE_COLOR, TextureType::BaseColor, scene);
	material.normalsIndex = loadTexture(aiMat, aiTextureType_NORMALS, TextureType::BaseColor, scene);
	material.metallicRoughnessIndex = loadTexture(aiMat, aiTextureType_METALNESS, TextureType::BaseColor, scene);
	material.ambientOcclusionIndex = loadTexture(aiMat, aiTextureType_AMBIENT_OCCLUSION, TextureType::BaseColor, scene);

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

struct TraverseParams {
	const aiScene *aiScene;
	Scene &scene;
	SizeType staticVertexOffset;
	SizeType movingVertexOffset;
	SizeType staticIndexOffset;
	SizeType movingIndexOffset;
	SceneLoaderFlags flags;
};

struct HashCmpStr {
	size_t operator()(char const *str) const {
		int p = 131;
		int hash = 0;
		while (*str) {
			hash = (hash * p) + (*str);
			str++;
		}

		return hash & (0x7FFFFFFF);
	}
};

struct StrEq {
	bool operator()(const char *a, const char *b) const {
		return strcmp(a, b) == 0;
	}
};

using NodeName2ParentMap = Map < const char *, aiNode *, HashCmpStr, StrEq>;

void traverseSkeleton(aiNode *node, const aiScene *aiScene, NodeName2ParentMap &childToParent, aiNode *parent, int tabs) {
	for (int i = 0; i < tabs; ++i) {
		printf(" ");
	}
	printf("%s:\n", node->mName.C_Str());
	for (int i = 0; i < node->mNumMeshes; ++i) {
		aiMesh *m = aiScene->mMeshes[node->mMeshes[i]];
		for (int i = 0; i < tabs; ++i) {
			printf(" ");
		}
		printf("  (%s) ", m->mName.C_Str());
		for (int j = 0; j < m->mNumBones; ++j) {
			printf("[%s] ", m->mBones[j]->mName.C_Str());
		}
		printf("\n");
	}

	++tabs;
	++tabs;
	++tabs;
	++tabs;
	childToParent[node->mName.C_Str()] = parent;
	for (int i = 0; i < node->mNumChildren; ++i) {
		traverseSkeleton(node->mChildren[i], aiScene, childToParent, node, tabs);
	}
	--tabs;
	--tabs;
	--tabs;
	--tabs;
}

void traverseAssimpScene(TraverseParams &params) {
	dassert(params.aiScene != nullptr);

	if (params.aiScene == nullptr) {
		return;
	}

	const aiScene *aiScene = params.aiScene;
	Scene &scene = params.scene;

	NodeName2ParentMap childToParent;
	using BoneName2IndexMap = Map < const char *, u8, HashCmpStr, StrEq>;
	BoneName2IndexMap boneNameToIndexInSkeleton;
	BoneName2IndexMap boneNameToSkeletonIndex;
	Map<SkeletonId, SkinnedModelNode*> skeletonIdToModel;

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

	if (aiScene->HasAnimations()) {
		traverseSkeleton(aiScene->mRootNode, aiScene, childToParent, nullptr, 0);
	}

	for (int i = 0; i < aiScene->mNumMeshes; ++i) {
		aiMesh *mesh = aiScene->mMeshes[i];
		SkeletonId skeletonId = INVALID_SKELETON_ID;
		AnimationSkeleton skeleton = {};
		
		const bool skinned = mesh->HasBones();
		if (skinned) {
			skeletonId = scene.addNewSkeleton();

			skeleton.numJoints = mesh->mNumBones;
			skeleton.joints = new Joint[skeleton.numJoints];
		}
		for (int j = 0; j < mesh->mNumBones; ++j) {
			aiBone *b = mesh->mBones[j];
			boneNameToIndexInSkeleton[b->mName.C_Str()] = j;
			boneNameToSkeletonIndex[b->mName.C_Str()] = skeletonId;

			dassert(childToParent[b->mName.C_Str()] != nullptr);
				
			auto parentIt = boneNameToIndexInSkeleton.find(childToParent[b->mName.C_Str()]->mName.C_Str());

			Joint &joint = skeleton.joints[j];
			joint.name = b->mName.C_Str();
			joint.inverseBindPose = *reinterpret_cast<Mat4 *>(&b->mOffsetMatrix);
			joint.parentJoint = (parentIt == boneNameToIndexInSkeleton.end() ? INVALID_JOINT_ID : parentIt->second);
		}

		const bool genTangents = mesh->HasTextureCoords(0) && ((params.flags & sceneLoaderFlags_overrideGenTangents) || !mesh->HasTangentsAndBitangents());

		ModelNode *model =  skinned ? new SkinnedModelNode : new ModelNode;
		if (skinned) {
			scene.getSkeleton(skeletonId) = skeleton;
			auto *sm = dynamic_cast<SkinnedModelNode*>(model);
			sm->skeleton = skeletonId;
			skeletonIdToModel[skeletonId] = sm;
		}

		SizeType &indexOffset = skinned ? params.movingIndexOffset : params.staticIndexOffset;
		SizeType &vertexOffset = skinned ? params.movingVertexOffset : params.staticVertexOffset;

		// Setup the mesh
		Mesh resMesh;
		resMesh.type = skinned ? MeshType::Skinned : MeshType::Static;
		resMesh.indexOffset = indexOffset;
		resMesh.numIndices = mesh->mNumFaces ? mesh->mNumFaces * mesh->mFaces[0].mNumIndices : 0;
		resMesh.mat = readMaterialDataForMesh(mesh, aiScene, params.scene);

		// Make sure the next mesh knows where its indices begin
		indexOffset += resMesh.numIndices;

		// save vertex data for the mesh in the global scene structure
		for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
			SkinnedVertex vertex;
			vertex.boneWeights = Vec3(-1.f);

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

			if (skinned) {
				scene.movingData.vertices.push_back(vertex);
			} else {
				scene.staticData.vertices.push_back(vertex);
			}
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
				if (skinned) {
					scene.movingData.indices.push_back(offsetIndex);
				} else {
					scene.staticData.indices.push_back(offsetIndex);
				}

				if (genTangents) {
					indicesMapping[face.mIndices[k]] = offsetIndex;
				}
			}

			// Generate normals and tangents if the mesh doesn't contain them
			const SizeType index = skinned ? scene.movingData.indices.size() - 3 : scene.staticData.indices.size() - 3;
			if (!mesh->HasNormals() || !mesh->HasTangentsAndBitangents() || (genTangents && !mesh->HasNormals())) {
				StaticVertex *v[3];
				v[0] = skinned ? &scene.movingData.vertices[scene.movingData.indices[index + 0]] : &scene.staticData.vertices[scene.staticData.indices[index + 0]];
				v[1] = skinned ? &scene.movingData.vertices[scene.movingData.indices[index + 1]] : &scene.staticData.vertices[scene.staticData.indices[index + 1]];
				v[2] = skinned ? &scene.movingData.vertices[scene.movingData.indices[index + 2]] : &scene.staticData.vertices[scene.staticData.indices[index + 2]];

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

		if (skinned) {
			Vector<u8> numBones(mesh->mNumVertices, 0);
			const SizeType meshVertexOffset = vertexOffset;

			const int offset = skeleton.numJoints;
			for (int j = 0; j < mesh->mNumBones; ++j) {
				aiBone *bone = mesh->mBones[j];

				// Update the vertex information for each vertex attached to this bone
				for (int k = 0; k < bone->mNumWeights; ++k) {
					aiVertexWeight &w = bone->mWeights[k];
					SkinnedVertex &v = scene.movingData.vertices[meshVertexOffset + w.mVertexId];

					u8 &numBonesInVert = numBones[w.mVertexId];
					v.boneIDs[numBonesInVert] = u8(j);
					if (numBonesInVert < 3) {
						v.boneWeights[numBonesInVert] = w.mWeight;
					}

					++numBonesInVert;
				}
			}
		}

		// Add the number of vertices we added to the global scene vertex buffer
		// so the next mesh's indices are offset correctly.
		vertexOffset += mesh->mNumVertices;

		// Finish up
		model->id = scene.nodes.size();
		scene.nodes.push_back(model);
	}

	using BoneNameToPoseMap = Map<const char *, JointPose, HashCmpStr, StrEq>;
	struct ParseAnimSample {
		double time;
		BoneNameToPoseMap poses;
	};
	struct ParseAnim {
		String name;
		Vector<ParseAnimSample> samples;
	};
	for (unsigned int i = 0; i < aiScene->mNumAnimations; ++i) {
		ParseAnim animation = {};
		AnimationId animId = scene.addNewAnimationClip();
		AnimationClip &clip = scene.getAnimationClip(animId);

		aiAnimation *anim = aiScene->mAnimations[i];
		animation.name = anim->mName.C_Str();

		SkeletonId skeletonId = INVALID_SKELETON_ID;

		// Skeleton of the animation
		u8 numJoints = 0;

		for (u32 j = 0; j < anim->mNumChannels; ++j) {
			aiNodeAnim *chan = anim->mChannels[j];
			if (j == 0) {
				// We need to somehow get the skeleton related to that animation
				const char *boneName = chan->mNodeName.C_Str();
				skeletonId = boneNameToSkeletonIndex[boneName];
				const AnimationSkeleton &s = scene.getSkeleton(skeletonId);
				numJoints = s.numJoints;
				animation.samples.resize(chan->mNumRotationKeys); // mNumPositionKeys == number of samples in the animation
			}

			dassert(chan->mNumRotationKeys == animation.samples.size());

			for (u32 k = 0; k < chan->mNumRotationKeys; ++k) {
				aiQuatKey &rotation = chan->mRotationKeys[k];

				JointPose pose = {};
				pose.rotation = Quat{ rotation.mValue.x, rotation.mValue.y, rotation.mValue.z, rotation.mValue.w };

				animation.samples[k].poses[chan->mNodeName.C_Str()] = pose;
				animation.samples[k].time = rotation.mTime;
			}
		}

		// Populate our animation structure
		clip.name = anim->mName.C_Str();
		clip.skeleton = skeletonId;
		clip.duration = anim->mDuration / anim->mTicksPerSecond;
		clip.frameCount = animation.samples.size();
		clip.samples = new AnimationSample[clip.frameCount];
		for (int i = 0; i < clip.frameCount; ++i) {
			clip.samples[i].jointPoses = new JointPose[numJoints];
			
			auto it = animation.samples[i].poses.begin();
			while (it != animation.samples[i].poses.end()) {
				JointId joint = boneNameToIndexInSkeleton[it->first];
				clip.samples[i].jointPoses[joint] = it->second;
				++it;
			}
		}

		auto *model = skeletonIdToModel[skeletonId];
		model->animations.push_back(animId);
	}
}

// TODO: make own importer implementation. Should be able to import .obj, gltf2 files.
SceneLoaderError loadScene(const WString &pathW, Scene &scene, SceneLoaderFlags flags) {
	Assimp::Importer &importer = getAssimpImporter();

	String path;
	WStringConverter wstrConverter;
	path = wstrConverter.to_bytes(pathW);

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
		aiProcess_PopulateArmatureData |
		aiProcess_ConvertToLeftHanded
	);
	if (!assimpScene || assimpScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !assimpScene->mRootNode) {
		return SceneLoaderError::InvalidScene;
	}

	SizeType vertexOffset = 0;
	SizeType indexOffset = 0;
	TraverseParams params = {
		assimpScene,
		scene,
		scene.staticData.vertices.size(),
		scene.movingData.vertices.size(),
		scene.staticData.indices.size(),
		scene.movingData.indices.size(),
		flags
	};
	traverseAssimpScene(params);

	return SceneLoaderError::Success;
}
