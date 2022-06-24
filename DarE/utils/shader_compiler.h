#pragma once

#include "utils/defines.h"
#include "d3d12/includes.h"

struct IDxcBlob;

namespace Dar {

struct ShaderCompilerResult {
	IDxcBlob *binary;
};

enum class ShaderType {
	Vertex,
	Pixel,
	Hull,
	Mesh,
	Geometry,
	Compute
};

namespace ShaderCompiler {

ShaderCompilerResult compileFromFile(const WString &filename);

ShaderCompilerResult compileFromSource(const char *src, const WString &name, const WString &outputDir, ShaderType type);

} // namespace ShaderCompiler

} // namespace Dar