#pragma once

#include "dar/utils/defines.h"

namespace Dar {

struct ImageHeader {
	Vector<SizeType> mipOffsets; ///< Offsets in data for each mip-level
	String filename;
	SizeType size = 0; ///< Size in bytes of the image data. DirectX loads it as is, since it's already mipped and compressed
	int width = 0;
	int height = 0;
	int ncomp = 0;
	int mipMapCount = 0;
};

// TODO: We could do any processing here:
// loading/generating mips, BCn/other compression, etc.
// Also store that metadata in the image header and write it to file in the serde module.
// TODO: see Compressonator SDK
struct ImageData {
	ImageHeader header = {};
	uint8_t *data = nullptr;

	void deinit();

	bool loadFromStream(std::ifstream& ifs, SizeType pos);
};

} // namespace Dar
