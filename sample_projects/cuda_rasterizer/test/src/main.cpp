#include "d3d12_cuda_rasterizer.h"

#include "cuda_manager.h"
#include "cuda_drawable.h"

#include <chrono>
#include "imgui.h"

double deltaTime = 0.;
double totalTime = 0.;
double fps = 0.;

void timeIt() {
	using std::chrono::duration_cast;
	using HRC = std::chrono::high_resolution_clock;

	static constexpr double SECONDS_IN_NANOSECOND = 1e-9;
	static UINT64 frameCount = 0;
	static double elapsedTime = 0.0;
	static HRC clock;
	static HRC::time_point t0 = clock.now();

	HRC::time_point t1 = clock.now();
	deltaTime = (t1 - t0).count() * SECONDS_IN_NANOSECOND;
	elapsedTime += deltaTime;
	totalTime += deltaTime;

	++frameCount;
	t0 = t1;

	if (elapsedTime > 1.0) {
		fps = frameCount / elapsedTime;

		frameCount = 0;
		elapsedTime = 0.0;
	}
}

void updateFrame(CudaRasterizer &rasterizer, void *state) {
	Mesh *mesh = reinterpret_cast<Mesh*>(state);

	timeIt();
	Vec4 a = { 0.1f, 0.3f, 1.f, 1.f };
	
	rasterizer.setClearColor(a);
	rasterizer.clearRenderTarget();
	rasterizer.clearDepthBuffer();

	// Render the image with CUDA
	mesh->draw(rasterizer);
}

void drawUI() {
	ImGui::Begin("FPS Counter");
	ImGui::Text("FPS: %.2f", fps);
	ImGui::End();
}

int main(int argc, char **argv) {
	Mesh *mesh = new Mesh("res\\obj\\head.obj", "BasicShader");

	CudaRasterizer rasterizer(Vector<String>{"data\\basic_shader.ptx"}, "CudaRasterizer", 1280, 720);
	if (!rasterizer.isInitialized()) {
		return 1;
	}

	rasterizer.setUpdateFramebufferCallback(updateFrame, reinterpret_cast<void*>(mesh));
	rasterizer.setImGuiCallback(drawUI);

	// TODO: Add perspective camera.
	// TODO: Add texture loading and uploading as UAV.
	// TODO: Make Drawable read texture coordinates and do basic sampling inside rasterizer.cu

	const int result = rasterizer.run();

	return result;
}
