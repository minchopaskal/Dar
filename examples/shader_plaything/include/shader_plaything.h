#pragma once

#include "framework/app.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "graphics/d3d12/resource_handle.h"

#include "ImGuiColorTextEdit/TextEditor.h"

#include <filesystem>

struct IDxcBlob;

struct ShaderPlaything : Dar::App {
	ShaderPlaything(UINT width, UINT height, const String &windowTitle);

private:
	// Inherited via D3D12App
	bool initImpl() override;
	void deinit() override;
	void update() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;

	Dar::FrameData& getFrameData();
	void beginFrame() override;
	void endFrame() override;

	//bool addRenderPassFromName(const String &shaderName, const String &displayName, int numRenderTargets);
	//bool addRenderPassFromSource(const char *code, const String &displayName, int numRenderTargets);
	bool createRenderPass(const String &name);

	// Topological ordering of render graph.
	void prepareRenderGraph();

	int loadPipelines();

	void resetFrameCount() {
		frameCountOffset = renderer.getNumRenderedFrames() + 1;
	}

private:
	using Super = Dar::App;

	using RenderPassId = SizeType;
	static constexpr RenderPassId INVALID_PASS_ID = RenderPassId(-1);
	struct RenderPass {
		String name = "";
		String shaderName = "";
		String shaderSource = "";
		Vector<Dar::RenderTarget> renderTextures;
		Vector<RenderPassId> dependancies;
		Vector<Dar::TextureResource> textures;
		Vector<String> textureDescs;
		TextEditor textEdit;
		bool compiled = false;
		bool needUpdate = false;
		Dar::HeapHandle texturesHeap; ///< Heap of the memory holding the textures' data

		// TODO: This doesn't currently work.
		// We need to rebuild the pipeline after adding
		// more render targets.
		void addRenderTexture(const ShaderPlaything &app);

		void addResourceTexture(const String &path);

		bool uploadTextures(Dar::UploadHandle);
	};

	struct ConstantData {
		int width;
		int height;
		unsigned int frame;
		float delta;
		float time;
		int hasOutput;
		Vec2 seed;
	};

	float timePassed = 0.f;

	Dar::Renderer renderer;
	Dar::FramePipeline framePipeline;

	Dar::FrameData frameData[Dar::FRAME_COUNT];
	Dar::DataBufferResource constData[Dar::FRAME_COUNT];
	Vector<RenderPass*> renderPasses;
	Vector<RenderPassId> renderGraph;
	RenderPassId outputPassId{ INVALID_PASS_ID };

	std::filesystem::path shadersPath;

	char buffer[4096];
	char nameBuffer[32];
	bool inputTextActive = false;
	bool updatePipelines = false;
	bool pauseAnimation = false;

	UINT64 freezeFrameCount = 0;

	/// Each time we set a new output make sure to restart the frameCounter as seen by the shaders.
	/// This way any frameCount dependant operations will be correct.
	UINT64 frameCountOffset = 0;

	static inline int shaderCount{ 0 };
};
