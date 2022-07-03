#pragma once

#include "utils/defines.h"
#include "d3d12/includes.h"

struct IDxcBlob;

namespace Dar {

enum class ShaderType {
	Vertex,
	Pixel,
	Hull,
	Mesh,
	Geometry,
	Compute
};

namespace ShaderCompiler {

bool compileFromFile(const WString &filename, const WString &outputDir, ShaderType type);

bool compileFromSource(const char *src, const WString &name, const WString &outputDir, ShaderType type);

} // namespace ShaderCompiler

} // namespace Dar