#include "img_data.h"

#include <fstream>

namespace Dar {

void ImageData::deinit() {
	delete data;
	data = nullptr;

	header = {};
}

bool ImageData::loadFromStream(std::ifstream& ifs, SizeType pos) {
	auto size = header.size;
	if (size == 0) {
		LOG_FMT(Error, "Trying to load image %s with 0 size!", header.filename.c_str());
		return false;
	}

	data = new uint8_t[size];

	ifs.seekg(pos, std::ios::beg);
	SizeType offset = 0;
	while (size > 0 && !ifs.eof()) {
		SizeType chunk = size > 4096 ? 4096 : size;
		ifs.read(reinterpret_cast<char*>(data + offset), chunk);
		size -= chunk;
		offset += chunk;
	}

	return true;
}

} // namespace Dar
