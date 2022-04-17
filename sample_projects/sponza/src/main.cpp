#include "sponza.h"

#include "async/async.h"
#include "utils/profile.h"

int main(int argc, char **argv) {
	DAR_OPTICK_APP("Sponza");

	Sponza app(1280, 720, "Sponza");
	if (!app.init()) {
		return 1;
	}
	
	return app.run();
}
