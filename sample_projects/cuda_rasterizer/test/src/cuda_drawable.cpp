#include "cuda_drawable.h"
#include "d3d12_cuda_rasterizer.h"

#include <fstream>
#include <sstream>

Mesh::Mesh(const char *objFilePath, const char *shaderName) {
	shader = shaderName;

	// read model from obj file
	std::ifstream in;
	in.open(objFilePath, std::ifstream::in);
	if (in.fail()) {
		return;
	}
	Vector<Vec4> positions;
	Vector<Vec3> normals;
	Vector<Vec2> uvs;
	Map<int, int> vertToUV;

	std::string line;
	while (!in.eof()) {
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "v ")) {
			iss >> trash;
			Vec4 v;
			for (int i = 0; i < 3; i++) {
				iss >> v.data[i];
			}
			v.w = 1.f;

			// Out "in" vector is (0.0, 0.0, 1.0) instead of (0.0, 0.0, -1.0)
			// so reverse the .z
			v.z *= -1.f;

			positions.push_back(v);
		} else if (!line.compare(0, 3, "vn ")) {
			iss >> trash >> trash;
			Vec3 normal;
			iss >> normal.x >> normal.y >> normal.z;
			normals.push_back(normal);
		} else if (!line.compare(0, 3, "vt ")) {
			iss >> trash >> trash;
			Vec2 uv;
			iss >> uv.x >> uv.y;
			uvs.push_back(uv);
		} else if (!line.compare(0, 2, "f ")) {
			int vertIdx, uvIdx, itrash;
			iss >> trash;
			while (iss >> vertIdx >> trash >> uvIdx >> trash >> itrash) {
				vertIdx--; // in wavefront obj all indices start at 1, not zero
				indices.push_back(vertIdx);
				--uvIdx;
				if (vertToUV.find(vertIdx) == vertToUV.end()) {
					vertToUV[vertIdx] = uvIdx;
				}
			}
		}
	}

	massert(positions.size() == normals.size());

	// Resolve vertices
	for (int i = 0; i < positions.size(); ++i) {
		Vertex v = { positions[i], normals[i], uvs[vertToUV[i]] };
		geometry.push_back(v);
	}
}

void Mesh::draw(CudaRasterizer &renderer) const {
	constexpr SizeType verticesInTriangle = 3;
	
	renderer.setUseDepthBuffer(true);
	renderer.setCulling(cullType_backface);

	renderer.setShaderProgram(shader);
	renderer.setVertexBuffer(geometry.data(), geometry.size());

	if (indices.empty()) {
		indices.resize(geometry.size() * verticesInTriangle);
		for (int i = 0; i < indices.size(); ++i) {
			indices[i] = i;
		}
	}
	renderer.setIndexBuffer(indices.data(), indices.size());

	const CUresult err = cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, indices.size() / verticesInTriangle + 200);
	if (err != CUDA_SUCCESS) {
		return;
	}

	renderer.drawIndexed(static_cast<unsigned int>(indices.size() / verticesInTriangle));
}

void Scene::draw(CudaRasterizer &renderer) const {
	for (const auto &mesh : meshes) {
		mesh.draw(renderer);
	}
}
