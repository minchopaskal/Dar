#pragma once

#include "img_data.h"
#include "dar/graphics/d3d12/includes.h"

#include "dxcapi.h"

namespace Dar {
	
namespace TxLib {

const SizeType INVALID_IMG_DATA_POS = SizeType(-1);

/// Header written at the beginning of a serialized textures data file(txlib).
/// ImageHeader objects are placed in a Vector as their respective image data
/// is ordered in the file.
/// imgDataStartPos is the position in the txlib where the texture data begins.
struct Header {
	Vector<ImageHeader> headers;
	SizeType imgDataStartPos = INVALID_IMG_DATA_POS;
};

/// Create a file named textures.txlib inside outputDir containing the given image data.
/// @param imgs Images to be serialized
/// @param outputDir Where to put the output file
/// @return true on success, false otherwise
bool serializeTextureDataToFile(const Vector<String> &imgs, const fs::path &outputDir);

Header readHeader(const fs::path &txLibFile);

} // namespace TxLib

namespace ShaderCompiler {

enum class ShaderType {
	Vertex = 0,
	Pixel,
	Hull,
	Mesh,
	Geometry,
	Compute,

	COUNT
};


struct CompiledShader {
	ComPtr<IDxcBlob> blob;
	String name;
};

Optional<CompiledShader> compileFromSource(const char *source, SizeType srcLen, const String &basename, const Vector<WString> includeDirs, ShaderType type);

/// @brief Given a shader base name generates a single file containing all compiled shaders.
/// @param basename of shaders. Shaders should follow the following template - basename_{vs,ps,etc}.hlsl
/// @param outputDir Directory to output the compiled shaders. Output file will always be `shaders.shlib`
/// @param truncateFile Delete contents of `shaders.shlib` if it aldready exists.
/// @return true of it manages to compile the shaders, false otherwise
bool compileShaderAsBlob(const String &basename, const String &outputDir, bool truncateFile);

/// @brief Same as compileShaderAsBlob but looks for all hlsl files.
bool compileFolderAsBlob(const String &shaderFolder, const String &outputDir);

/// @brief Given a shader blob file returns a vector of blobs containing compiled shaders together with their types.
/// @return empty vector if the file was invalid or there were no shaders in the file.
Vector<CompiledShader> readBlob(const String &filename);

} // namespace ShaderCompiler

} // namespace Dar
