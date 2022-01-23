#include "sponza.h"

int main(int argc, char **argv) {
	Sponza app(1280, 720, "Sponza");
	if (!app.init()) {
		return 1;
	}
	
	if (!app.loadAssets()) {
		return 2;
	}
	
	return app.run();
}
