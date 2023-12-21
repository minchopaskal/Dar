#pragma once

#include "utils/defines.h"

struct Scene;

enum class SceneLoaderError : int {
	Success = 0,
	UnsupportedExtention,
	InvalidScene,
	InvalidScenePath,
	CorruptSceneFile,
};

enum SceneLoaderFlags : int {
	sceneLoaderFlags_none = 0,
	sceneLoaderFlags_overrideGenTangents = 1 << 0,
};

SceneLoaderError loadScene(
	const String &path,
	Scene &scene,
	SceneLoaderFlags flags = sceneLoaderFlags_none
);
