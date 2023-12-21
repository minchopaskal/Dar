#include "shader_plaything.h"

int main() {
	ShaderPlaything app(1280, 720, "Shader Plaything");
	if (!app.init()) {
		return 1;
	}
	
	return app.run();
}
