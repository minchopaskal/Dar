#include "d3d12_cuda_rasterizer.h"

#include "cuda_manager.h"

int main(int argc, char **argv) {
	initializeCUDAManager("data\\rasterizer.ptx");

	CudaRasterizer app(1280, 720, "CudaRasterizer");
	if (!app.init()) {
		return 1;
	}

	if (!app.loadAssets()) {
		return 1;
	}
	
	int result = app.run();

	deinitializeCUDAManager();

	return result;
}
