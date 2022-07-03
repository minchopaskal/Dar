#include "shader_toy.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "asset_manager/asset_manager.h"
#include "framework/app.h"
#include "d3d12/resource_manager.h"
#include "d3d12/pipeline_state.h"
#include "utils/shader_compiler.h"
#include "utils/utils.h"

#include <fstream>
#include <sstream>

ShaderToy::ShaderToy(UINT width, UINT height, const String &windowTitle) : Dar::App(width, height, windowTitle.c_str()) {
	memset(buffer, 0, 4096);
	memset(nameBuffer, 0, 32);
}

int ShaderToy::initImpl() {
	setUseImGui();

	constData[0].init(sizeof(ConstantData), 1);
	constData[1].init(sizeof(ConstantData), 1);

	addRenderPass(L"render_pass", "Render Pass 1", 1);
	outputPassId = 0;

	return loadPipelines();
}

void ShaderToy::deinit() {
	flush();
	Super::deinit();
}

void ShaderToy::update() {
	auto &resManager = Dar::getResourceManager();

	if (updatePipelines) {
		loadPipelines();
		updatePipelines = false;
	}

	ConstantData cd = { };
	cd.width = width;
	cd.height = height;
	cd.frame = frameCount;
	cd.delta = deltaTime;
	cd.hasOutput = (outputPassId >= 0 && outputPassId < renderPasses.size());

	const int frameIdx = renderer.getBackbufferIndex();

	Dar::UploadHandle uploadHandle = resManager.beginNewUpload();
	resManager.uploadBufferData(uploadHandle, constData[frameIdx].getHandle(), &cd, sizeof(ConstantData));
	resManager.uploadBuffers();

	Dar::FrameData &fd = frameData[frameIdx];
	fd.addConstResource(constData[frameIdx].getHandle(), 0);
	
	const int numRPs = renderGraph.size();
	for (int i = 0; i < numRPs; ++i) {
		for (int j : renderPasses[renderGraph[i]]->dependancies) {
			for (auto &rt : renderPasses[j]->renderTextures) {
				fd.addTextureResource(rt.getTextureResource(frameIdx), i);
			}
		}
		fd.addRenderCommand(Dar::RenderCommand::drawInstanced(3, 1, 0, 0), i);
	}

	if (outputPassId >= 0 && outputPassId < renderPasses.size()) {
		for (auto &rt : renderPasses[outputPassId]->renderTextures) {
			fd.addTextureResource(rt.getTextureResource(frameIdx), numRPs);
		}
	}

	fd.addRenderCommand(Dar::RenderCommand::drawInstanced(3, 1, 0, 0), numRPs);
}

void ShaderToy::drawUI() {
	ImGui::SetNextWindowPos({ 0, 0 });
	
	ImGui::Begin("FPS", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("%.2f", getFPS());
	ImGui::End();

	ImGui::Begin("Render Passes", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		static int depRP = 0;

		for (int i = 0; i < renderPasses.size(); ++i) {
			auto &rp = renderPasses[i];

			ImGui::Text("%d: %s", i, rp->name.c_str());

			ImGui::SameLine();
			if (ImGui::Button(i == outputPassId ? "Remove as output" : "Set as output")) {
				outputPassId = (i == outputPassId ? INVALID_PASS_ID : i);
				updatePipelines = true;
			}
			
			ImGui::SameLine();
			if (ImGui::Button("Add dependancy")) {
				depRP = i;
				ImGui::OpenPopup("add_dep_popup");
			}

			ImGui::SameLine();
			if (ImGui::Button("Remove dependancy")) {
				depRP = i;
				ImGui::OpenPopup("rem_dep_popup");
			}

			ImGui::Text("Dependancies:");
			for (auto j : rp->dependancies) {
				ImGui::Text("\t%s", renderPasses[j]->name.c_str());
			}

			ImGui::Separator();
		}

		if (ImGui::BeginPopup("add_dep_popup")) {
			auto &deps = renderPasses[depRP]->dependancies;
			for (int i = 0; i < renderPasses.size(); ++i) {
				// slow but whatever
				if (i == depRP || std::find(deps.begin(), deps.end(), i) != deps.end()) {
					continue;
				}

				if (ImGui::Button(renderPasses[i]->name.c_str())) {
					deps.push_back(i);
					if (std::find(renderGraph.begin(), renderGraph.end(), depRP) != renderGraph.end()) {
						updatePipelines = true;
					}
					ImGui::CloseCurrentPopup();
				}
			}
			ImGui::EndPopup();
		}

		if (ImGui::BeginPopup("rem_dep_popup")) {
			auto &deps = renderPasses[depRP]->dependancies;
			for (int i = 0; i < deps.size(); ++i) {
				if (ImGui::Button(renderPasses[deps[i]]->name.c_str())) {
					deps.erase(deps.begin() + i);
					if (std::find(renderGraph.begin(), renderGraph.end(), depRP) != renderGraph.end()) {
						updatePipelines = true;
					}
					ImGui::CloseCurrentPopup();
				}
			}
			ImGui::EndPopup();
		}
	ImGui::End();

	ImGui::Begin("Shader Sources");
	if (ImGui::Button("(+) Shader")) {
		static int shaderCount = 0;
		addRenderPass(String("Shader") + std::to_string(shaderCount++));
	}

	if (ImGui::BeginTabBar("Files tab")) {
		for (int i = 0; i < renderPasses.size(); ++i) {
			auto &rp = renderPasses[i];
			if (ImGui::BeginTabItem(rp->name.c_str())) {
				rp->textEdit.Render("Editor");

				auto it = std::find(renderGraph.begin(), renderGraph.end(), i);

				if (ImGui::Button("Recomiple")) {
					LOG(InfoFancy, "RECOMPILE PRESSED");
					static const char *vertexSource = "#include \"res\\shaders\\screen_quad.hlsli\"";
					rp->shaderSource = rp->textEdit.GetText();
					bool res = Dar::ShaderCompiler::compileFromSource(vertexSource, rp->shaderName, L".\\res\\shaders", Dar::ShaderType::Vertex);
					res = res && Dar::ShaderCompiler::compileFromSource(rp->shaderSource.data(), rp->shaderName, L".\\res\\shaders", Dar::ShaderType::Pixel);
					// Only recompile the pipeline if this render pass in the render graph structure
					if (res && it != renderGraph.end()) {
						updatePipelines = true;
					}
				}

				ImGui::EndTabItem();
			}
		}

		ImGui::EndTabBar();
	}
	ImGui::End();
}

void ShaderToy::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = dmath::max(1u, w);
	this->height = dmath::max(1u, h);

	flush();

	renderer.resizeBackBuffers();
	for (auto &rp : renderPasses) {
		for (auto &rt : rp->renderTextures) {
			rt.resizeRenderTarget(width, height);
		}
	}
}

void ShaderToy::onKeyboardInput(int key, int action) {
	// TODO: fix input not detected
	return;

	if (inputTextActive) {
		return;
	}

	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	auto &rs = renderer.getSettings();

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		rs.vSyncEnabled = !rs.vSyncEnabled;
	}
}

void ShaderToy::onMouseScroll(double xOffset, double yOffset) { }

Dar::FrameData &ShaderToy::getFrameData() {
	return frameData[renderer.getBackbufferIndex()];
}

bool ShaderToy::addRenderPass(const WString &shaderName, const String &displayName, int numRenderPasses) {
	std::ifstream ifs(Dar::getAssetFullPath((shaderName + L"_ps.hlsl").c_str(), Dar::AssetType::Shader));
	if (!ifs.good()) {
		return false;
	}

	renderPasses.emplace_back(nullptr);

	auto *&rpp = renderPasses.back();
	rpp = new RenderPass();

	RenderPass &rp = *rpp;
	rp.name = displayName;
	rp.shaderName = shaderName;
	rp.shaderSource = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
	
	rp.textEdit.SetText(rp.shaderSource);

	Dar::TextureInitData texInitData = {};
	texInitData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
	texInitData.width = width;
	texInitData.height = height;
	for (int i = 0; i < numRenderPasses; ++i) {
		rp.renderTextures.emplace_back();
		auto &rt = rp.renderTextures.back();
		rt.init(texInitData, Dar::FRAME_COUNT);
		rt.setName(L"RenderTexture[" + std::to_wstring(renderPasses.size() - 1) + L"]");
	}

	return true;
}

bool ShaderToy::addRenderPass(const char *code, const String &displayName, int numRenderPasses) {
	auto shaderName = L"shader" + std::to_wstring(__COUNTER__);
	bool res = Dar::ShaderCompiler::compileFromSource(code, shaderName, L".\\res\\shaders", Dar::ShaderType::Pixel);

	if (!res) {
		return false;
	}

	return addRenderPass(shaderName, displayName, numRenderPasses);
}

bool ShaderToy::addRenderPass(const String &name) {
	renderPasses.emplace_back();

	RenderPass *&rpp = renderPasses.back();
	rpp = new RenderPass;

	RenderPass &rp = *rpp;
	rp.name = name;
	rp.shaderName = L"shader" + std::to_wstring(__COUNTER__);
	rp.shaderSource = "";
	rp.textEdit.SetText("");

	// Add a single render texture
	rp.renderTextures.emplace_back();
	auto &rt = rp.renderTextures.back();

	Dar::TextureInitData texInitData = {};
	texInitData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
	texInitData.width = getWidth();
	texInitData.height = getHeight();
	rt.init(texInitData, Dar::FRAME_COUNT);

	return true;
}

void ShaderToy::prepareRenderGraph() {
	if (outputPassId < 0 || outputPassId >= renderPasses.size()) {
		return;
	}

	renderGraph.clear();

	Stack<RenderPassId> s;
	Set<RenderPassId> visited;
	s.push(outputPassId);
	while (!s.empty()) {
		auto id = s.top();

		if (renderPasses[id]->dependancies.empty()) {
			renderGraph.push_back(id);
			s.pop();
			continue;
		}

		for (auto d : renderPasses[id]->dependancies) {
			if (visited.find(d) == visited.end()) {
				continue;
			}

			s.push(d);
			break;
		}

		renderGraph.push_back(id);
		s.pop();
	}
}

int ShaderToy::loadPipelines() {
	prepareRenderGraph();

	CD3DX12_STATIC_SAMPLER_DESC sampler{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR };

	Dar::PipelineStateDesc psDesc = {};
	Dar::RenderPassDesc renderPassDesc = {};

	for (int i = 0; i < renderGraph.size(); ++i) {
		RenderPass *rp = renderPasses[renderGraph[i]];
		WString shaderName = rp->shaderName;

		psDesc.shaderName = shaderName;
		psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
		psDesc.numRenderTargets = 1;
		psDesc.renderTargetFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		psDesc.staticSamplerDesc = &sampler;
		psDesc.numConstantBufferViews = 1;
		psDesc.cullMode = D3D12_CULL_MODE_NONE;
		psDesc.numTextures = rp->dependancies.size();

		renderPassDesc = {};
		renderPassDesc.setPipelineStateDesc(psDesc);
		for (auto &rt : rp->renderTextures) {
			renderPassDesc.attach(Dar::RenderPassAttachment::renderTarget(&rt));
		}
		renderer.addRenderPass(renderPassDesc);
	}

	psDesc = {};
	psDesc.shaderName = L"screen_quad";
	psDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	psDesc.numRenderTargets = 1;
	psDesc.renderTargetFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	psDesc.staticSamplerDesc = &sampler;
	psDesc.cullMode = D3D12_CULL_MODE_NONE;
	psDesc.numTextures = !renderPasses.empty();
	psDesc.numConstantBufferViews = 1;

	renderPassDesc = {};
	renderPassDesc.setPipelineStateDesc(psDesc);
	renderPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderer.addRenderPass(renderPassDesc);

	renderer.compilePipeline();

	prepareRenderGraph();

	return true;
}
