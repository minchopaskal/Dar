#include "d3d12_cuda_rasterizer.h"

#include "cuda_manager.h"

int main(int argc, char **argv) {
	

	CudaRasterizer app(Vector<String>{"data\\basic_shader.ptx"}, "CudaRasterizer", 1280, 720);
	if (!app.init()) {
		return 2;
	}

	if (!app.loadScene("")) {
		return 1;
	}
	
	int result = app.run();

	return result;
}
