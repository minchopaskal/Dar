#include "d3d12_tex_cube.h"

int main(int argc, char **argv) {
	D3D12TexturedCube app(1280, 720, "Textured Cube");
	app.init();
	app.loadAssets();
	
	return app.run();
}
