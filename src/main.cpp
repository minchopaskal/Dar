#include "d3d12_hello_triangle.h"

int main(int argc, char **argv) {
	D3D12HelloTriangle app(1280, 720, "Hello Triangle");
	app.init();
	app.loadAssets();
	
	return app.run();
}
