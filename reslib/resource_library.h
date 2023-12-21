#pragma once

#include "async/async.h"
#include "serde.h"

namespace Dar {

class ResourceLibrary {
public:
	// Texture resources
	void LoadTextureData();
	ImageData getImageData(const String &imageName) const;

	// Shader resources
	void LoadShaderData();
	void addShader(ShaderCompiler::CompiledShader shader);
	IDxcBlob *getShader(const String &name) const;

	// TODO: other resources...

private:
	ResourceLibrary() = default;

	struct ImagePos {
		ImageData data;
		SizeType pos;
	};

	Map<String, ComPtr<IDxcBlob>> shaders;
	mutable Map<String, ImagePos> imageName2Data;

	SpinLock initializing;
	bool initTextureData = false;
	bool initShaderData = false;

	friend void initResourceLibrary();
};

void initResourceLibrary();
ResourceLibrary& getResourceLibrary();
void deinitResourceLibrary();

} // namespace Dar
