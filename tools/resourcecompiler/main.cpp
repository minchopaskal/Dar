#include <cstdio>
#include <filesystem>

#include "reslib/serde.h"

namespace fs = std::filesystem;

bool iterateTexturesDir(fs::path p, const fs::path &outputDir) {
	Vector<String> imgPaths;
	LOG_FMT(Info, "Compiling textures folder %s...", p.string().c_str());
	for (auto t : fs::directory_iterator{ p }) {
		if (t.is_regular_file()) {
			imgPaths.push_back(t.path().string());
		}
	}

	if (!Dar::TxLib::serializeTextureDataToFile(imgPaths, outputDir)) {
		LOG_FMT(Error, "Failed to create %stextures.txlib file!", outputDir.string().c_str());
		return false;
	}

	LOG_FMT(Info, "Successfully created %stextures.txlib", outputDir.string().c_str());

	return true;
}

bool compileShaders(fs::path dir, const fs::path &outputDir) {
	LOG_FMT(Info, "Compiling shaders from %s...", dir.string().c_str());
	if (Dar::ShaderCompiler::compileFolderAsBlob(dir.string(), outputDir.string())) {
		LOG_FMT(Info, "Written %sshaders.shlib!", outputDir.string().c_str());
		return true;
	}

	LOG(Error, "Failed to compile shaders!");
	return false;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		LOG_FMT(
			Error,
			"Usage: %s <res_dir> <lib_output_dir> [resource_type]\n"
			"\tOptional resource_type: shaders, textures\n",
			"\tSearches in res_dir for the following folders: scenes, shaders, textures",
			argv[0]
		);

		exit(1);
	}

	String inputDir = argv[1];
	auto outputDir = fs::path(argv[2]);
	auto resource_type = (argc > 3 ? Optional<String>(argv[3]) : std::nullopt);

	for (auto p : fs::directory_iterator{ inputDir }) {
		if (!p.is_directory()) {
			continue;
		}

		auto path = p.path();

		if (path.filename() == "textures" && (!resource_type.has_value() || *resource_type == "textures")) {
			if (!iterateTexturesDir(path, outputDir)) {
				return 1;
			}
		}

		if (path.filename() == "shaders" && (!resource_type.has_value() || *resource_type == "shaders")) {
			if (!compileShaders(path, outputDir)) {
				return 1;
			}
		}

		// TODO: else...
	}

	return 0;
}