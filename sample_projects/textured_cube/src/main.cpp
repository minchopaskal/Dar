#include "d3d12_tex_cube.h"

int main(int argc, char **argv) {
	D3D12TexturedCube app(1280, 720, "Textured Cube");
	if (!app.init()) {
		return 1;
	}
	
	if (!app.loadAssets()) {
		return 1;
	}
	
	return app.run();
}
