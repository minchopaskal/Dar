#include "resource_library.h"

#include <fstream>

namespace Dar {

void ResourceLibrary::LoadTextureData() {
	if (initTextureData) {
		return;
	}
	auto lock = initializing.lock();
	if (initTextureData) {
		return;
	}

	auto header = TxLib::readHeader(L".\\res\\textures\\textures.txlib");

	if (header.headers.empty() || header.imgDataStartPos == TxLib::INVALID_IMG_DATA_POS) {
		return;
	}

	SizeType offset = 0;
	for (auto &imgh : header.headers) {
		ImageData data{ .header = imgh, .data = nullptr };
		imageName2Data.insert({ imgh.filename, ImagePos{.data = data, .pos = header.imgDataStartPos + offset } });

		offset += imgh.size;
	}

	initTextureData = true;
}

ImageData ResourceLibrary::getImageData(const String &imageName) const {
	if (auto it = imageName2Data.find(imageName); it != imageName2Data.end()) {
		auto& img = it->second.data;
		if (img.data != nullptr) {
			return img;
		}

		std::ifstream ifs(".\\res\\textures\\textures.txlib", std::ios::binary | std::ios::in);
		if (!ifs.good()) {
			LOG(Error, "Failed to load textures.txlib file!");
			return ImageData{};
		}
		
		if (!img.loadFromStream(ifs, it->second.pos)) {
			return ImageData{};
		}

		return img;
	} else {
		LOG_FMT(Error, "Unknown texture file %s!", imageName.c_str());
	}

	return ImageData{};
}

void ResourceLibrary::LoadShaderData() {
	if (initShaderData) {
		return;
	}
	auto lock = initializing.lock();
	if (initShaderData) {
		return;
	}

	auto compiled = ShaderCompiler::readBlob(".\\res\\shaders\\shaders.shlib");

	for (auto shader : compiled) {
		addShader(shader);
	}

	initShaderData = true;
}

void ResourceLibrary::addShader(ShaderCompiler::CompiledShader shader) {
	if (auto it = shaders.find(shader.name); it != shaders.end()) {
		it->second = shader.blob;
	} else {
		shaders.insert({ shader.name, shader.blob });
	}
}

IDxcBlob *ResourceLibrary::getShader(const String &name) const {
	if (auto it = shaders.find(name); it != shaders.end()) {
		return it->second.Get();
	}

	return nullptr;
}

static ResourceLibrary *reslib = nullptr;

void initResourceLibrary() {
	if (reslib == nullptr) {
		reslib = new ResourceLibrary;
	}
}

ResourceLibrary& getResourceLibrary() {
	return *reslib;
}

void deinitResourceLibrary() {
	delete reslib;
	reslib = nullptr;
}

} // namespace Dar
