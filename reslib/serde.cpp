#include "serde.h"

#include <fstream>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#include "nvtt/nvtt.h"

namespace Dar {

namespace TxLib {

String GetLastErrorAsString() {
	DWORD errorMessageID = ::GetLastError();
	if (errorMessageID == 0) {
		return {};
	}

	LPSTR messageBuffer = nullptr;

	size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

	String message(messageBuffer, size);

	LocalFree(messageBuffer);

	return message;
}

constexpr uint32_t HEADER_END = 0xFAFAFAFA;

struct OutputHandler : nvtt::OutputHandler {
	ImageData &buffer;
	SizeType bufSize;
	SizeType offset;

	OutputHandler(ImageData &buffer, int estSize) : buffer(buffer), bufSize(estSize), offset(0) {}

	// Inherited via OutputHandler
	void beginImage(int size, int /*width*/, int /*height*/, int /*depth*/, int /*face*/, int miplevel) override {
		if (miplevel == 0) {
			bufSize = std::max(bufSize, static_cast<SizeType>(size * 1.35)); // wiggle-room, as mip-maped size is usually 1.33x the size of the image
			buffer.data = new uint8_t[bufSize];
		}

		buffer.header.mipOffsets.push_back(offset);
	}

	bool writeData(const void* data, int size) override {
		if (offset + size > bufSize) {
			return false;
		}

		memcpy(reinterpret_cast<void*>(buffer.data + offset), data, static_cast<SizeType>(size));
		offset += static_cast<SizeType>(size);

		return true;
	}

	void endImage() override {
	}
};

bool serializeTextureDataToFile(const Vector<String> &imgPaths, const fs::path &outputDir) {
	if (imgPaths.empty()) {
		LOG(Error, "No image data to serialize!");
		return false;
	}

	if (!fs::exists(outputDir)) {
		auto mkdir = fs::absolute(outputDir);
		fs::create_directories(mkdir);
	}

	auto outPath = std::filesystem::path(outputDir) / L"textures.txlib";
	outPath = std::filesystem::absolute(outPath);

	const bool stdioOldSyncFlag = std::ios_base::sync_with_stdio(false);

	auto fstreamFlags = std::ios::binary | std::ios::out | std::ios::trunc;
	std::ofstream ofs(outPath, fstreamFlags);
	if (!ofs.good()) {
		LOG_FMT(Error, "Failed to open %s!", outPath.string().c_str());
		return false;
	}

	nvtt::Context nvttCtx;

	// TODO: we need different format per texture type.
	// TODO: Also maybe normal maps need to be renormalized after resizing?
	nvtt::CompressionOptions compressionOpts;
	compressionOpts.setFormat(nvtt::Format_BC7);

	Vector<ImageData> imgs;
	for (auto& imgPath : imgPaths) {
		ImageData img;

		bool hasAlpha;
		nvtt::Surface nvttImg;
		if (!nvttImg.load(imgPath.c_str(), &hasAlpha)) {
			LOG_FMT(Error, "Failed to load %s! Skipping...", imgPath.c_str());
			continue;
		}

		img.header.filename = fs::path(imgPath).filename().string();
		img.header.width = nvttImg.width();
		img.header.height = nvttImg.height();
		img.header.ncomp = 4; // TODO: is this true
		img.header.mipMapCount = nvttImg.countMipmaps();

		LOG_FMT(Info, "Compiling texture file %s...", img.header.filename.c_str());

		auto estSize = nvttCtx.estimateSize(nvttImg, img.header.mipMapCount, compressionOpts);

		OutputHandler outputHandler{ img, estSize };

		nvtt::OutputOptions outputOpts;
		outputOpts.setOutputHandler(&outputHandler);

		bool success = true;
		for (int i = 0; i < img.header.mipMapCount; ++i) {
			if (!nvttCtx.compress(nvttImg, 0 /* face */, i, compressionOpts, outputOpts)) {
				LOG_FMT(Error, "Failed to compress %s", imgPath.c_str());
				success = false;
				break;
			}

			nvttImg.toLinearFromSrgb();
			if (hasAlpha) {
				nvttImg.premultiplyAlpha();
			}

			nvttImg.buildNextMipmap(nvtt::MipmapFilter_Box);

			nvttImg.demultiplyAlpha();
			nvttImg.toSrgb();
		}

		if (!success) {
			continue;
		}

		img.header.size = outputHandler.offset;

		imgs.push_back(img);
	}

	for (auto &img : imgs) {
		dassert(img.header.mipMapCount == static_cast<int>(img.header.mipOffsets.size()));

		auto nameSz = static_cast<uint32_t>(img.header.filename.size() * sizeof(String::value_type));
		ofs.write(reinterpret_cast<char *>(&nameSz), sizeof(uint32_t));
		ofs.write(reinterpret_cast<const char*>(img.header.filename.data()), nameSz);

		ofs.write(reinterpret_cast<const char *>(&img.header.size), sizeof(SizeType));
		ofs.write(reinterpret_cast<const char*>(&img.header.width), sizeof(int));
		ofs.write(reinterpret_cast<const char*>(&img.header.height), sizeof(int));
		ofs.write(reinterpret_cast<const char*>(&img.header.ncomp), sizeof(int));
		ofs.write(reinterpret_cast<const char*>(&img.header.mipMapCount), sizeof(int));
		for (SizeType i = 0; i < img.header.mipOffsets.size(); ++i) {
			ofs.write(reinterpret_cast<const char*>(&img.header.mipOffsets[i]), sizeof(SizeType));
		}
	}

	uint32_t headerEnd = HEADER_END;
	ofs.write(reinterpret_cast<char *>(&headerEnd), sizeof(uint32_t));

	for (auto &img : imgs) {
		SizeType dataSize = img.header.size;
		SizeType offset = 0;
		while (dataSize > 0) {
			SizeType chunk = dataSize < 4*1024 ? dataSize : 4*1024;
			ofs.write(reinterpret_cast<const char *>(img.data) + offset, chunk);
			if (ofs.fail()) {
				LOG_FMT(Error, "Failed to serialize %ls! Error: %s(%d)", img.header.filename.c_str(), GetLastErrorAsString().c_str(), GetLastError());
				return false;
			}

			dataSize -= chunk;
			offset += chunk;
		}
	}

	std::for_each(imgs.begin(), imgs.end(), [](ImageData& img) { STBI_FREE((void*)img.data); });

	ofs.close();

	std::ios_base::sync_with_stdio(stdioOldSyncFlag);

	return true;
}

Header readHeader(const fs::path &txLibFile) {
	std::ifstream ifs(txLibFile, std::ios::in | std::ios::binary);
	if (!ifs.good()) {
		LOG_FMT(Error, "Could not open %s!", txLibFile.string().c_str());
		return Header{};
	}

	Header result;
	SizeType headerSize = 0;
	while (!ifs.eof()) {
		uint32_t nameSz;
		ifs.read(reinterpret_cast<char *>(&nameSz), sizeof(uint32_t));

		if (nameSz == HEADER_END) {
			headerSize += sizeof(uint32_t); // header end
			break;
		}

		ImageHeader header;
		header.filename.resize(nameSz / sizeof(String::value_type));
		ifs.read(reinterpret_cast<char*>(header.filename.data()), nameSz);
		ifs.read(reinterpret_cast<char*>(&header.size), sizeof(SizeType));
		ifs.read(reinterpret_cast<char*>(&header.width), sizeof(int));
		ifs.read(reinterpret_cast<char*>(&header.height), sizeof(int));
		ifs.read(reinterpret_cast<char*>(&header.ncomp), sizeof(int));
		ifs.read(reinterpret_cast<char*>(&header.mipMapCount), sizeof(int));
		for (int i = 0; i < header.mipMapCount; ++i) {
			SizeType offset;
			ifs.read(reinterpret_cast<char*>(&offset), sizeof(SizeType));
			header.mipOffsets.push_back(offset);
		}

		headerSize += sizeof(uint32_t) + nameSz + sizeof(SizeType) + 4 * sizeof(int) + header.mipMapCount * sizeof(SizeType);

		result.headers.push_back(header);
	}

#pragma warning(suppress: 4189)
	const SizeType streampos = ifs.tellg();
	dassert(streampos == headerSize);

	result.imgDataStartPos = headerSize;

	if (result.headers.empty()) {
		return Header{ };
	}

	return result;
}

} // namespace TxLib

namespace ShaderCompiler {

String shaderTypeToStr(ShaderType type) {
	switch (type) {
	case ShaderType::Vertex:
		return "vs";
	case ShaderType::Pixel:
		return "ps";
	default:
		// IMPLEMENT ME
		break;
	}

	return "";
}

WString shaderTypeToWStr(ShaderType type) {
	switch (type) {
	case ShaderType::Vertex:
		return L"vs";
	case ShaderType::Pixel:
		return L"ps";
	default:
		// IMPLEMENT ME
		break;
	}

	return L"";
}

bool outputBlobToFile(const Vector<CompiledShader> &shaders, const String &outputDir, bool truncateFile) {
	if (!std::filesystem::exists(outputDir)) {
		auto mkdir = std::filesystem::absolute(outputDir);
		std::filesystem::create_directories(mkdir);
	}

	auto outPath = std::filesystem::path(outputDir) / L"shaders.shlib";
	outPath = std::filesystem::absolute(outPath);

	auto fstreamFlags = std::ios::binary | std::ios::out;
	if (truncateFile) {
		fstreamFlags |= std::ios::trunc;
	} else {
		fstreamFlags |= std::ios::app;
	}
	std::ofstream ofs(outPath, fstreamFlags);
	if (!ofs.good()) {
		LOG_FMT(Error, "Failed to write %s, error: %s", outPath.string().c_str(), strerror(errno));
		return false;
	}

	if (truncateFile) {
		const uint32_t header = 0xFADE00BE;
		ofs.write(reinterpret_cast<const char *>(&header), sizeof(uint32_t));
	}

	for (auto &shader : shaders) {
		auto nameSz = uint32_t(shader.name.size() * sizeof(String::value_type));
		ofs.write(reinterpret_cast<char *>(&nameSz), sizeof(uint32_t));
		ofs.write(reinterpret_cast<const char *>(shader.name.data()), nameSz);

		// Let's hope our blobs don't get bigger than 4GB
		uint32_t sz = static_cast<uint32_t>(shader.blob->GetBufferSize());
		ofs.write(reinterpret_cast<char *>(&sz), sizeof(uint32_t));
		ofs.write(reinterpret_cast<char *>(shader.blob->GetBufferPointer()), shader.blob->GetBufferSize());
	}

	ofs.close();

	return true;
}

bool compileFolderAsBlob(const String &shaderFolder, const String &outputDir) {
	Set<String> basenames;

	for (auto &entry : std::filesystem::directory_iterator(shaderFolder)) {
		if (!entry.is_regular_file()) {
			continue;
		}

		auto filename = entry.path().filename().string();
		auto sv = std::string_view{ filename };

		auto extPos = sv.find_last_of('.');
		if (extPos == std::string::npos) {
			continue;
		}
		auto ext = sv.substr(extPos + 1);
		sv = sv.substr(0, extPos);
		if (ext != "hlsl") {
			continue;
		}

		extPos = sv.find_last_of('_');
		if (extPos == std::string::npos) {
			continue;
		}
		auto base = sv.substr(0, extPos);
		ext = sv.substr(extPos + 1);
		if (ext == "vs" || ext == "ps" /* || TODO */) {
			basenames.insert(filename.substr(0, extPos));
		}
	}

	bool truncateFile = true;
	for (auto &basename : basenames) {
		auto p = std::filesystem::path(shaderFolder) / basename;
		if (!compileShaderAsBlob(p.string(), outputDir, truncateFile)) {
			// TODO: delete generated file?
			return false;
		}
		truncateFile = false;
	}

	return true;
}

Optional<CompiledShader> compileFromSource(const char *src, SizeType srcLen, const String &name, const Vector<WString> includeDirs, ShaderType type) {
	ComPtr<IDxcCompiler3> compiler;
	DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf()));

	ComPtr<IDxcUtils> utils;
	DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(utils.GetAddressOf()));

	ComPtr<IDxcIncludeHandler> includeHandler;
	utils->CreateDefaultIncludeHandler(includeHandler.GetAddressOf());

	ComPtr<IDxcBlobEncoding> source;
	utils->CreateBlob(src, static_cast<uint32_t>(srcLen), CP_UTF8, source.GetAddressOf());

	Vector<WString> argsStr;
	for (auto &includeDir : includeDirs) {
		argsStr.push_back(L"-I");
		argsStr.push_back(includeDir);
	}
	argsStr.push_back(L"-E");
	argsStr.push_back(L"main");
	argsStr.push_back(L"-T");
	argsStr.push_back(shaderTypeToWStr(type) + L"_6_6");
	argsStr.push_back(L"-Qstrip_debug");
	argsStr.push_back(L"-Qstrip_reflect");
	argsStr.push_back(L"-no-warnings");

	Vector<LPCWSTR> args;
	for (auto &arg : argsStr) {
		args.push_back(arg.c_str());
	}

	DxcBuffer sourceBuffer;
	sourceBuffer.Ptr = source->GetBufferPointer();
	sourceBuffer.Size = source->GetBufferSize();
	sourceBuffer.Encoding = 0;

	ComPtr<IDxcResult> compileResult;
	compiler->Compile(&sourceBuffer, args.data(), uint32_t(args.size()), includeHandler.Get(), IID_PPV_ARGS(compileResult.GetAddressOf()));

	ComPtr<IDxcBlobUtf8> errors;
	compileResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(errors.GetAddressOf()), nullptr);
	if (errors && errors->GetStringLength() > 0) {
		LOG_FMT(Error, "DXC {Error(%s): %s", name.c_str(), (char *)errors->GetBufferPointer());
		return std::nullopt;
	}

	CompiledShader result = {};
	if (compileResult->HasOutput(DXC_OUT_OBJECT)) {
		result.name = name + "_" + shaderTypeToStr(type);
		RETURN_ON_ERROR_FMT(
			compileResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(result.blob.GetAddressOf()), nullptr),
			std::nullopt,
			"Failed to get shader %s output!", name.c_str()
		);
	}

	return result;
}

bool compileShaderAsBlob(const String &basename, const String &outputDir, bool truncateFile) {
	

	Vector <CompiledShader> result;
	for (int i = 0; i < static_cast<int>(ShaderType::COUNT); ++i) {
		auto shaderType = static_cast<ShaderType>(i);
		const auto p = std::filesystem::absolute(basename + "_" + shaderTypeToStr(shaderType) + ".hlsl");
		auto shaderName = p.stem().string();

		if (!std::filesystem::exists(p) || !std::filesystem::is_regular_file(p)) {
			continue;
		}

		std::ifstream ifs(p.c_str(), std::ios::in | std::ios::ate);
		if (!ifs.good()) {
			continue;
		}

		const auto size = ifs.tellg();
		auto srcMemblock = std::make_unique<char[]>(size);
		ifs.seekg(0, std::ios::beg);
		ifs.read(srcMemblock.get(), size);
		ifs.close();

		auto include_dir = p.parent_path().wstring();
		auto compiled = compileFromSource(srcMemblock.get(), size, basename, {include_dir}, shaderType);
		if (compiled.has_value()) {
			result.push_back(*compiled);
		}
	}

	if (result.empty()) {
		return false;
	}

	outputBlobToFile(result, outputDir, truncateFile);

	return true;
}

// TODO: error ifs error checking
Vector<CompiledShader> readBlob(const String &filename) {
	auto p = std::filesystem::path(filename);

	auto res = Vector<CompiledShader>{};

	if (!std::filesystem::exists(p)) {
		return res;
	}

	std::ifstream ifs(p, std::ios::in | std::ios::binary);
	if (!ifs.good()) {
		return res;
	}

	uint32_t header;
	ifs.read(reinterpret_cast<char *>(&header), sizeof(uint32_t));
	if (header != 0xFADE00BE) {
		return res;
	}

	ComPtr<IDxcUtils> utils;
	DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(utils.GetAddressOf()));

	while (!ifs.eof()) {
		CompiledShader shader{};

		uint32_t nameSz;
		ifs.read(reinterpret_cast<char *>(&nameSz), sizeof(uint32_t));
		if (ifs.eof()) {
			break;
		}

		shader.name.resize(nameSz / sizeof(String::value_type));
		ifs.read(reinterpret_cast<char *>(shader.name.data()), nameSz);

		uint32_t blob_size;
		ifs.read(reinterpret_cast<char *>(&blob_size), sizeof(uint32_t));

		auto memblock = std::make_unique<char[]>(blob_size);
		ifs.read(memblock.get(), blob_size);

		ComPtr<IDxcBlobEncoding> blob;
		utils->CreateBlob((void *)memblock.get(), blob_size, DXC_CP_ACP, blob.GetAddressOf());
		blob.As<IDxcBlob>(&shader.blob);

		res.push_back(shader);
	}

	std::ignore = ifs.get();
	dassert(ifs.eof());

	return res;
}

} // namespace ShaderCompiler

} // namespace Dar
