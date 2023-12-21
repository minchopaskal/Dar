#include "hello_triangle.h"

int main() {
	HelloTriangle app(1280, 720, "Hello Triangle");
	if (!app.init()) {
		return 1;
	}
	
	return app.run();
}
