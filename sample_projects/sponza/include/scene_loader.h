#pragma once

#include "d3d12_scene.h"

enum class SceneLoaderError : int {
	Success = 0,
	UnsupportedExtention,
	InvalidScene
};

SceneLoaderError loadScene(
	const String &path,
	Scene &scene
);
