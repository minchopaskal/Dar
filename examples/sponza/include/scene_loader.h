#pragma once

#include "utils/defines.h"

struct Scene;

enum class SceneLoaderError : int {
	Success = 0,
	UnsupportedExtention,
	InvalidScene,
	InvalidScenePath,
	InvalidPath,
	CorruptSceneFile,
};

SceneLoaderError loadScene(
	const String &path,
	Scene &scene
);
