#include "d3d12_cuda_rasterizer.h"

#include "cuda_manager.h"
#include "cuda_drawable.h"

#include <chrono>
#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"

double DELTA_TIME = 0.;
double TOTAL_TIME = 0.;
double FPS = 0.;

struct Camera {
	Camera() :
		fov(90.0),
		yaw(0.0),
		pitch(0.0),
		front(Vec3(0.f, 0.f, 1.f)),
		right(Vec3(1.f, 0.f, 0.f)),
		up(Vec3(0.f, 1.f, 0.f)),
		position(Vec3(.0f, .0f, -2.f)) {
	}

	[[nodiscard]] Mat4 getViewMatrix() const {
		return dmath::lookAt(position + front, position, up);
	}

	double fov;
	double yaw, pitch;
	Vec3 front, right, up;
	Vec3 position;
} CAMERA;

struct State {
	Mesh *mesh = nullptr;
	CUDADefaultBuffer texBufferCuda;
	TextureSampler sampler = {};
	bool textureUploaded = false;
};

void timeIt() {
	using std::chrono::duration_cast;
	using Hrc = std::chrono::high_resolution_clock;

	static constexpr double seconds_in_nanosecond = 1e-9;
	static UINT64 frameCount = 0;
	static double elapsedTime = 0.0;
	static Hrc::time_point t0 = Hrc::now();

	const Hrc::time_point t1 = Hrc::now();
	DELTA_TIME = static_cast<double>((t1 - t0).count()) * seconds_in_nanosecond;
	elapsedTime += DELTA_TIME;
	TOTAL_TIME += DELTA_TIME;

	++frameCount;
	t0 = t1;

	if (elapsedTime > 1.0) {
		FPS = static_cast<double>(frameCount) / elapsedTime;

		frameCount = 0;
		elapsedTime = 0.0;
	}
}

void updateFrame(CudaRasterizer &rasterizer, void *state) {
	auto st = static_cast<State*>(state);
	Mesh *mesh = st->mesh;
	TextureSampler &sampler= st->sampler;

	timeIt();
	Vec4 a = { 0.1f, 0.3f, 1.f, 1.f };
	
	CUDAError err = rasterizer.setClearColor(a);
	err = rasterizer.clearRenderTarget();
	err = rasterizer.clearDepthBuffer();

	constexpr double speed = 30.;
	Mat4 modelMat(1.f);
	modelMat = modelMat.rotate(Vec3(0.f, 1.f, 0.f), static_cast<float>(speed * TOTAL_TIME));
	err = rasterizer.setUavBuffer(&modelMat, sizeof(Mat4), 0);

	Mat4 viewMat = CAMERA.getViewMatrix();
	err = rasterizer.setUavBuffer(&viewMat, sizeof(Mat4), 1);

	Mat4 projectionMat = dmath::perspective(static_cast<float>(CAMERA.fov), rasterizer.getWidth() / static_cast<float>(rasterizer.getHeight()), 0.0001f, 10000.f);
	err = rasterizer.setUavBuffer(&projectionMat, sizeof(Mat4), 2);

	Mat4 normalMat = modelMat.inverse();
	normalMat = normalMat.transpose();
	err = rasterizer.setUavBuffer(&normalMat, sizeof(Mat4), 3);

	if (!st->textureUploaded) {
		st->texBufferCuda.initialize(sampler.width * sampler.height * sampler.numComp);
		st->texBufferCuda.upload(sampler.data);
		stbi_image_free(sampler.data);
		sampler.data = reinterpret_cast<unsigned char*>(st->texBufferCuda.handle());
		err = rasterizer.setUavBuffer(&sampler, sizeof(TextureSampler), 4);
		st->textureUploaded = true;
	}

	if (err.hasError()) {
		return;
	}

	// Render the image with CUDA
	mesh->draw(rasterizer);
}

void drawUI() {
	ImGui::Begin("FPS Counter");
	ImGui::Text("FPS: %.2f", FPS);
	ImGui::End();
}

int main(int argc, char **argv) {
	auto shaders = Vector<String>{ "data\\basic_shader.ptx" };
	CudaRasterizer rasterizer(shaders, "CudaRasterizer", 1280, 720);
	if (!rasterizer.isInitialized()) {
		return 1;
	}

	const auto mesh = new Mesh("res\\obj\\head.obj", "BasicShader");

	TextureSampler sampler = {};
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
