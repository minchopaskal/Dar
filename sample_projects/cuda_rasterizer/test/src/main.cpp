#include "d3d12_cuda_rasterizer.h"

#include "cuda_manager.h"
#include "cuda_drawable.h"

#include <chrono>
#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"

double deltaTime = 0.;
double totalTime = 0.;
double fps = 0.;

struct Camera {
	Camera() :
		FOV(90.0),
		yaw(0.0),
		pitch(0.0),
		front(Vec3(0.f, 0.f, 1.f)),
		right(Vec3(1.f, 0.f, 0.f)),
		up(Vec3(0.f, 1.f, 0.f)),
		position(Vec3(.0f, 0.f, .0f)) {
	}

	Mat4 getViewMatrix() const {
		return dmath::lookAt(position + front, position, up);
	}

	double FOV;
	double yaw, pitch;
	Vec3 front, right, up;
	Vec3 position;
} camera;

struct State {
	Mesh *mesh = nullptr;
	CUDADefaultBuffer texBufferCUDA;
	TextureSampler sampler;
	bool textureUploaded = false;
};

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
	State *st = reinterpret_cast<State*>(state);
	Mesh *mesh = st->mesh;
	TextureSampler &sampler= st->sampler;

	timeIt();
	Vec4 a = { 0.1f, 0.3f, 1.f, 1.f };
	
	rasterizer.setClearColor(a);
	rasterizer.clearRenderTarget();
	rasterizer.clearDepthBuffer();

	const float speed = 30.f;
	Mat4 modelMat(1.f);
	modelMat = modelMat.rotate(Vec3(0.f, 1.f, 0.f), speed * totalTime);
	rasterizer.setUavBuffer(&modelMat, sizeof(Mat4), 0);

	Mat4 viewMat = camera.getViewMatrix();
	rasterizer.setUavBuffer(&viewMat, sizeof(Mat4), 1);

	Mat4 projectionMat = dmath::perspective((float)camera.FOV, rasterizer.getWidth() / (float)rasterizer.getHeight(), 0.0001f, 10000.f);
	rasterizer.setUavBuffer(&projectionMat, sizeof(Mat4), 2);

	Mat4 normalMat = modelMat.inverse();
	normalMat = normalMat.transpose();
	rasterizer.setUavBuffer(&normalMat, sizeof(Mat4), 3);

	if (!st->textureUploaded) {
		st->texBufferCUDA.initialize(sampler.width * sampler.height * sampler.numComp);
		st->texBufferCUDA.upload(sampler.data);
		stbi_image_free(sampler.data);
		sampler.data = reinterpret_cast<unsigned char*>(st->texBufferCUDA.handle());
		rasterizer.setUavBuffer(&sampler, sizeof(TextureSampler), 4);
		st->textureUploaded = true;
	}

	// Render the image with CUDA
	mesh->draw(rasterizer);
}

void drawUI() {
	ImGui::Begin("FPS Counter");
	ImGui::Text("FPS: %.2f", fps);
	ImGui::End();
}

int main(int argc, char **argv) {
	CudaRasterizer rasterizer(Vector<String>{"data\\basic_shader.ptx"}, "CudaRasterizer", 1280, 720);
	if (!rasterizer.isInitialized()) {
		return 1;
	}

	Mesh *mesh = new Mesh("res\\obj\\head.obj", "BasicShader");

	TextureSampler sampler;
	stbi_set_flip_vertically_on_load(true);
	sampler.data = stbi_load("res\\tex\\head.tga", &sampler.width, &sampler.height, &sampler.numComp, 0);

	State state;
	state.mesh = mesh;
	state.sampler = sampler;

	rasterizer.setUpdateFramebufferCallback(updateFrame, reinterpret_cast<void*>(&state));
	rasterizer.setImGuiCallback(drawUI);

	const int result = rasterizer.run();

	return result;
}
