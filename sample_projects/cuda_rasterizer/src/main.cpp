#include "d3d12_cuda_rasterizer.h"

#include "cuda_manager.h"

int main(int argc, char **argv) {
	initializeCUDAManager(Vector<String>{ "data\\rasterizer.ptx", "data\\basic_shader.ptx" }, true);

	CudaRasterizer app(1280, 720, "CudaRasterizer");
	if (!app.init()) {
		return 2;
	}

	if (!app.loadScene("")) {
		return 1;
	}
	
	int result = app.run();

	deinitializeCUDAManager();

	return result;
}
