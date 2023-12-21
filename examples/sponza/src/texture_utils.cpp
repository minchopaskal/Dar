#include "texture_utils.h"
#include "reslib/resource_library.h"

bool uploadTextureData(
	Vector<TextureDesc> &textureDescs,
	Dar::UploadHandle uploadHandle,
	Vector<Dar::TextureResource> &textures,
	Dar::HeapHandle &texturesHeap,
	bool forceNoMips
) {
	SizeType numTextures = textureDescs.size();

	std::for_each(
		textures.begin(),
		textures.end(),
		[](Dar::TextureResource &texResource) {
			texResource.deinit();
		}
	);

	textures.resize(numTextures);

	auto &reslib = Dar::getResourceLibrary();
	auto &resManager = Dar::getResourceManager();
	Vector<Dar::ImageData> texData(numTextures);
	Vector<Dar::ResourceInitData> resInitDatas(numTextures);
	for (int i = 0; i < numTextures; ++i) {
		const TextureDesc &tex = textureDescs[i];

		Dar::ImageData &td = texData[i];
		td = reslib.getImageData(fs::path(tex.path).string());

		if (forceNoMips) {
			td.header.mipMapCount = 1;
		}

		// Load some default texture
		if (td.header.width <= 0 || td.header.height <= 0 || td.header.ncomp != 4) {
			td.header.filename = "DEFAULT";
			td.header.width = 1;
			td.header.height = 1;
			td.header.ncomp = 1;
			td.deinit();
			td.data = new uint8_t[4];
			memset(td.data, 0xFF00FFFF, sizeof(int)); // magenta
		}

		char textureName[32] = "";
		snprintf(textureName, 32, "Texture[%d]", i);

		Dar::TextureInitData texInitData = {};
		texInitData.width = td.header.width;
		texInitData.height = td.header.height;
		texInitData.mipLevels = td.header.mipMapCount;
		// TODO: We don't need BC7 for normal maps, etc
		texInitData.format = DXGI_FORMAT_BC7_UNORM;

		Dar::ResourceInitData &resInitData = resInitDatas[i];
		resInitData.init(Dar::ResourceType::TextureBuffer);
		resInitData.textureData = texInitData;
		resInitData.name = textureName;
	}

	resManager.createHeap(resInitDatas, texturesHeap);

	if (texturesHeap == INVALID_HEAP_HANDLE) {
		LOG(Error, "Failed to create textures heap!");
		return false;
	}

	for (int i = 0; i < numTextures; ++i) {
		textures[i].init(resInitDatas[i].textureData, Dar::TextureResourceType::ShaderResource, resInitDatas[i].name, texturesHeap);
		std::ignore = textures[i].upload(uploadHandle, texData[i]);
	}

	return true;
}
