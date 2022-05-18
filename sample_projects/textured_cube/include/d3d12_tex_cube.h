#pragma once

#include "framework/app.h"
#include "framework/camera.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "d3d12/pipeline_state.h"

#include "fps_camera_controller.h"

struct D3D12TexturedCube : Dar::App {
	D3D12TexturedCube(UINT width, UINT height, const String &windowTitle);


private:
	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	void onMouseMove(double xPos, double yPos) override;
	Dar::FrameData& getFrameData() override;
	
	int loadAssets();
	bool resizeDepthBuffer();

private:
	using Super = Dar::App;

	static constexpr int numTextures = 1;
	struct TexturedCubePassArgs {
		Dar::Renderer &renderer;
		Dar::TextureResource *textureHandles;
		int numTextures;
	} renderPassArgs = { renderer, textures, numTextures };

	enum class ProjectionType {
		Perspective,
		Orthographic
	} projectionType = ProjectionType::Perspective;

	// Vertex buffer
	Dar::VertexBuffer vertexBuffer;
	Dar::IndexBuffer indexBuffer;
	Dar::DepthBuffer depthBuffer;

	// MVP matrix
	Dar::ResourceHandle mvpBufferHandle[Dar::FRAME_COUNT];

	// Texture data
	Dar::TextureResource textures[numTextures];

	Dar::FrameData frameData[Dar::FRAME_COUNT];

	// viewport
	float aspectRatio;
	float orthoDim = 10.f;

	// Camera
	Dar::Camera cam;
	FPSCameraController camControl;
};
