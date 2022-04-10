#include "sponza.h"

#include "async/async.h"

int main(int argc, char **argv) {
	Sponza app(1280, 720, "Sponza");
	if (!app.init()) {
		return 1;
	}
	
	return app.run();
}
