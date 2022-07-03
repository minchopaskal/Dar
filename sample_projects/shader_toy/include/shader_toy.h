#pragma once

#include "framework/app.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "graphics/d3d12/resource_handle.h"

#include "ImGuiColorTextEdit/TextEditor.h"

struct IDxcBlob;

struct ShaderToy : Dar::App {
	ShaderToy(UINT width, UINT height, const String &windowTitle);

private:
	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	Dar::FrameData& getFrameData() override;

	bool addRenderPass(const WString &shaderName, const String &displayName, int numRenderTargets);
	bool addRenderPass(const char *code, const String &displayName, int numRenderTargets);
	bool addRenderPass(const String &name);

	void prepareRenderGraph();

	int loadPipelines();

private:
	using Super = Dar::App;

	using RenderPassId = SizeType;
	static constexpr RenderPassId INVALID_PASS_ID = RenderPassId(-1);
	struct RenderPass {
		String name;
		WString shaderName;
		String shaderSource;
		Vector<Dar::RenderTarget> renderTextures;
		Vector<RenderPassId> dependancies;
		TextEditor textEdit;
	};

	struct ConstantData {
		int width;
		int height;
		int frame;
		float delta;
		int hasOutput;
	};

	Dar::FrameData frameData[Dar::FRAME_COUNT];
	Dar::DataBufferResource constData[Dar::FRAME_COUNT];
	Vector<RenderPass*> renderPasses;
	Vector<RenderPassId> renderGraph;
	RenderPassId outputPassId;

	char buffer[4096];
	char nameBuffer[32];
	bool inputTextActive = false;
	bool updatePipelines = false;
};
