#include "shader_toy.h"

int main(int argc, char **argv) {
	ShaderToy app(1280, 720, "Shader Toy");
	if (!app.init()) {
		return 1;
	}
	
	return app.run();
}
