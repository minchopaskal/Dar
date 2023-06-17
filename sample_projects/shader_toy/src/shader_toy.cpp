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
#include "utils/random.h"

#include "imguifiledialog/ImGuiFileDialog.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

// For loading the texture image
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

struct ImageData {
	void *data = nullptr;
	int width = 0;
	int height = 0;
	int ncomp = 0;
};

ImageData loadImage(const String &imgPath) {
	WString imgPathWStr(imgPath.begin(), imgPath.end());
	const WString fullPathWStr = imgPathWStr.c_str();
	const SizeType bufferLen = fullPathWStr.size() * sizeof(wchar_t) + 1;
	char *path = new char[bufferLen];
	stbi_convert_wchar_to_utf8(path, bufferLen, fullPathWStr.c_str());

	ImageData result = {};
	result.data = stbi_load(path, &result.width, &result.height, nullptr, 4);
	result.ncomp = 4;

	return result;
}

ShaderToy::ShaderToy(UINT width, UINT height, const String &windowTitle) : Dar::App(width, height, windowTitle.c_str()) {
	memset(buffer, 0, 4096);
	memset(nameBuffer, 0, 32);
	setNumThreads(2);
}

int ShaderToy::initImpl() {
	setUseImGui();

	constData[0].init(sizeof(ConstantData), 1);
	constData[1].init(sizeof(ConstantData), 1);

	auto homeDir = std::string{ getenv("USERPROFILE") };
	shadersPath = std::filesystem::path(homeDir) / "Documents" / "ShaderToy";
	std::filesystem::create_directory(shadersPath);

	return loadPipelines();
}

void ShaderToy::deinit() {
	flush();

	// Delete all the temporary shader files.
	for (auto &rp : renderPasses) {
		WString vsShaderName = rp->shaderName + WString(L"_vs.bin");
		WString psShaderName = rp->shaderName + WString(L"_ps.bin");

		std::filesystem::path p("res\\shaders\\");
		auto vsPath = p/vsShaderName;
		auto psPath = p/psShaderName;

		DeleteFileW(vsPath.c_str());
		DeleteFileW(psPath.c_str());
	}

	Super::deinit();
}

void ShaderToy::update() {
	const int frameIdx = renderer.getBackbufferIndex();
	Dar::FrameData &fd = frameData[frameIdx];

	if (updatePipelines) {
		loadPipelines();
		// Has to call begin frame on framedata again in order to update
		// the render passes count.
		// TODO: find a better way or refactor FrameData
		fd.beginFrame(renderer);
		updatePipelines = false;
	}

	const int frameCount = renderer.getNumRenderedFrames();
	timePassed += pauseAnimation ? 0.f : getDeltaTime();

	ConstantData cd = { };
	cd.width = width;
	cd.height = height;
	cd.frame = pauseAnimation ? freezeFrameCount : frameCount - frameCountOffset;
	cd.delta = getDeltaTime();
	cd.time = timePassed;
	cd.hasOutput = (outputPassId >= 0 && outputPassId < renderPasses.size());
	Dar::Random rand;
	cd.seed.x = rand.generateFlt(0.f, 1.f);
	cd.seed.y = rand.generateFlt(0.f, 1.f);

	auto &resManager = Dar::getResourceManager();
	Dar::UploadHandle uploadHandle = resManager.beginNewUpload();
	{
		resManager.uploadBufferData(uploadHandle, constData[frameIdx].getHandle(), &cd, sizeof(ConstantData));
		for (auto &rp : renderPasses) {
			rp->uploadTextures(uploadHandle);
		}
	}
	resManager.uploadBuffers();

	fd.addConstResource(constData[frameIdx].getHandle(), 0);
	
	const int numRPs = renderGraph.size();
	const int prevFrameIdx = frameIdx == 0 ? (Dar::FRAME_COUNT - 1) : ((frameIdx - 1) % Dar::FRAME_COUNT);
	for (int i = 0; i < numRPs; ++i) {
		auto &renderPass = renderPasses[renderGraph[i]];
		
		// Add the previous frame's rendered textures as a texture resource
		if (frameCount > 1) {
			for (auto &rt : renderPass->renderTextures) {
				fd.addTextureResource(rt.getTextureResource(prevFrameIdx), i);
			}
		}

		for (int j : renderPass->dependancies) {
			for (auto &rt : renderPasses[j]->renderTextures) {
				fd.addTextureResource(rt.getTextureResource(frameIdx), i);
			}
		}

		auto &textures = renderPass->textures;
		for (auto& texture : textures) {
			fd.addTextureResource(texture, i);
		}

		fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0), i);
	}

	if (outputPassId >= 0 && outputPassId < renderPasses.size()) {
		for (auto &rt : renderPasses[outputPassId]->renderTextures) {
			fd.addTextureResource(rt.getTextureResource(frameIdx), numRPs);
		}
	}

	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0), numRPs);
}

void ShaderToy::drawUI() {
	ImGui::SetNextWindowPos({ 0, 0 });
	
	ImGui::Begin("Shader toy", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("FPS: %.2f", getFPS());
		if (ImGui::Button("Restart")) {
			resetFrameCount();
			pauseAnimation = false;
		}
		ImGui::SameLine();
		if (ImGui::Button(pauseAnimation ? "Continue" : "Pause")) {
			pauseAnimation = !pauseAnimation;
			if (!pauseAnimation) {
				frameCountOffset = renderer.getNumRenderedFrames() - freezeFrameCount;
			} else {
				freezeFrameCount = renderer.getNumRenderedFrames() - frameCountOffset;
			}
		}

		ImGui::SameLine();
		if (pauseAnimation) {
			if (ImGui::Button("Step")) {
				++freezeFrameCount;
			}
		}
	ImGui::End();

	static int processedRP = 0;
	if (!renderPasses.empty()) {
	ImGui::Begin("Render Passes", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		const char *addPopupName = "add_dep_popup";
		const char *remPopupName = "rem_dep_popup";
		const char *addTextureName = "add_tex_popup";

		for (int i = 0; i < renderPasses.size(); ++i) {
			auto &rp = renderPasses[i];

			ImGui::Text("%s", rp->name.c_str());

			String outputButton = i == outputPassId ? "Remove as output" : "Set as output";
			outputButton += "##";
			outputButton += std::to_string(i);

			ImGui::SameLine();
			if (rp->compiled) {
				if (ImGui::Button(outputButton.c_str())) {
					resetFrameCount();
					outputPassId = (i == outputPassId ? INVALID_PASS_ID : i);
					updatePipelines = true;
				}
			}

			String addDepButton = "Add dependancy##";
			addDepButton += std::to_string(i);

			String remDepButton = "Remove dependancy##";
			remDepButton += std::to_string(i);

			String addRTButton = "Add render target##";
			addRTButton += std::to_string(i);

			String addTextureButton = "Add texture##";
			addTextureButton += std::to_string(i);

			ImGui::SameLine();
			if (ImGui::Button(addDepButton.c_str())) {
				processedRP = i;
				ImGui::OpenPopup(addPopupName);
			}

			ImGui::SameLine();
			if (ImGui::Button(remDepButton.c_str())) {
				processedRP = i;
				ImGui::OpenPopup(remPopupName);
			}

			ImGui::SameLine();
			if (ImGui::Button(addRTButton.c_str())) {
				ImGui::OpenPopup("Unsupported##1");
				// TODO:
				// renderPasses[i]->addRenderTexture(*this);
			}

			ImGui::SameLine();
			if (ImGui::Button(addTextureButton.c_str())) {
				processedRP = i;
				ImGui::OpenPopup(addTextureName);
			}

			ImGui::Text("Dependancies:");
			for (auto j : rp->dependancies) {
				ImGui::SameLine();
				ImGui::Text("\t%s", renderPasses[j]->name.c_str());
			}
			ImGui::Text("Num renderTargets: %d", renderPasses[i]->renderTextures.size());
			ImGui::Text("Num loaded textures: %d", renderPasses[i]->textures.size());

			ImGui::Separator();
		}

		if (ImGui::BeginPopup("Unsupported##1")) {
			ImGui::Text("WIP!");
			if (ImGui::Button("Ok")) {
				ImGui::CloseCurrentPopup();
			}
		}

		if (ImGui::BeginPopup(addPopupName)) {
			auto &deps = renderPasses[processedRP]->dependancies;
			if (renderPasses.size() == 1) {
				ImGui::Text("No render passes to add as dependancies!");
			}
			for (int j = 0; j < renderPasses.size(); ++j) {
				// slow but whatever - we don't expect a lot of shaders nevertheless.
				if (j == processedRP || std::find(deps.begin(), deps.end(), j) != deps.end()) {
					continue;
				}

				if (ImGui::Button(renderPasses[j]->name.c_str())) {
					deps.push_back(j);
					if (std::find(renderGraph.begin(), renderGraph.end(), processedRP) != renderGraph.end()) {
						updatePipelines = true;
					}
					ImGui::CloseCurrentPopup();
				}
			}
			ImGui::EndPopup();
		}

		if (ImGui::BeginPopup(remPopupName)) {
			auto &deps = renderPasses[processedRP]->dependancies;
			if (deps.empty()) {
				ImGui::Text("Render pass has no dependancies!");
			}
			for (int j = 0; j < deps.size(); ++j) {
				if (ImGui::Button(renderPasses[deps[j]]->name.c_str())) {
					deps.erase(deps.begin() + j);
					if (std::find(renderGraph.begin(), renderGraph.end(), processedRP) != renderGraph.end()) {
						updatePipelines = true;
					}
					ImGui::CloseCurrentPopup();
				}
			}
			ImGui::EndPopup();
		}

		if (ImGui::BeginPopup(addTextureName)) {
			auto path = shadersPath / "img.png";
			ImGuiFileDialog::Instance()->OpenDialog("ChooseTextureDlgKey", "Choose texture", ".png,.jpg,.jpeg", path.string());
		
			ImGui::EndPopup();
		}

	ImGui::End(); // Render passes window
	}

	ImGui::Begin("Shader Sources");
	if (ImGui::Button("(+) Shader")) {
		addRenderPass(String("Shader") + std::to_string(shaderCount++));
	}

	bool openCompileErrorPopup = false;
	inputTextActive = false;
	if (ImGui::BeginTabBar("Files tab")) {
		for (int i = 0; i < renderPasses.size(); ++i) {
			auto &rp = renderPasses[i];
			if (ImGui::BeginTabItem(rp->name.c_str())) {
				auto it = std::find(renderGraph.begin(), renderGraph.end(), i);
				String buttonIndex = std::to_string(i);

				String compileButton = String(rp->compiled ? "Recomiple##" : "Compile##") + buttonIndex;

				String loadFileButton = "Load from file##" + buttonIndex;

				String saveFileButton = "Save to file##" + buttonIndex;

				if (ImGui::Button(compileButton.c_str())) {
					static const char *vertexSource = "#include \"res\\shaders\\screen_quad.hlsli\"";
					rp->shaderSource = rp->textEdit.GetText();
					bool res = Dar::ShaderCompiler::compileFromSource(vertexSource, rp->shaderName, L".\\res\\shaders", Dar::ShaderType::Vertex);
					res = res && Dar::ShaderCompiler::compileFromSource(rp->shaderSource.data(), rp->shaderName, L".\\res\\shaders", Dar::ShaderType::Pixel);
					if (res) {
						rp->compiled = true;

						// Only recompile the pipeline if this render pass in the render graph structure
						if (it != renderGraph.end()) {
							updatePipelines = true;
							resetFrameCount();
						}
					} else {
						processedRP = i;
						openCompileErrorPopup = true;
					}
				}

				ImGui::SameLine();
				if (ImGui::Button(loadFileButton.c_str())) {
					auto path = shadersPath / (rp->name + ".hlsl");
					ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".hlsl", path.string());
					processedRP = i;
				}

				ImGui::SameLine();
				if (ImGui::Button(saveFileButton.c_str())) {
					auto path = shadersPath / (rp->name + ".hlsl");

					ImGuiFileDialog::Instance()->OpenDialog("SaveToFileDlgKey", "Save to File", ".hlsl", path.string());
					processedRP = i;
				}
				
				inputTextActive = inputTextActive || rp->textEdit.Render("Editor");

				ImGui::EndTabItem();
			}
		}

		ImGui::EndTabBar();
	}

	if (openCompileErrorPopup) {
		ImGui::OpenPopup("Error##compiler");
		openCompileErrorPopup = false;
	}

	bool popupOpen = true;
	if (ImGui::BeginPopupModal("Error##compiler", &popupOpen)) {
		auto &rp = renderPasses[processedRP];
		ImGui::Text("%s failed to compile!", rp->name.c_str());
		ImGui::EndPopup();
	}

	if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
		if (ImGuiFileDialog::Instance()->IsOk()) {
			String filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

			std::filesystem::path p(filePathName);

			if (std::filesystem::exists(p)) {
				auto &rp = renderPasses[processedRP];
				std::stringstream ss;
				for (const auto &line : Dar::generateFileLines(filePathName)) {
					ss << line << std::endl;
				}

				rp->shaderSource = ss.str();
				rp->textEdit.SetText(rp->shaderSource);
			}
		}

		ImGuiFileDialog::Instance()->Close();
	}

	if (ImGuiFileDialog::Instance()->Display("SaveToFileDlgKey")) {
		if (ImGuiFileDialog::Instance()->IsOk()) {
			String filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

			std::ofstream ofs(filePathName, std::ios::trunc);

			if (ofs.good()) {
				auto &rp = renderPasses[processedRP];
				rp->shaderSource = rp->textEdit.GetText();
				ofs << rp->shaderSource;
				ofs.close();
			}
		}

		ImGuiFileDialog::Instance()->Close();
	}

	if (ImGuiFileDialog::Instance()->Display("ChooseTextureDlgKey")) {
		if (ImGuiFileDialog::Instance()->IsOk()) {
			String filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

			renderPasses[processedRP]->addResourceTexture(filePathName);
		}

		ImGuiFileDialog::Instance()->Close();
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

bool ShaderToy::addRenderPass(const WString &shaderName, const String &displayName, int numRenderTargets) {
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
	for (int i = 0; i < numRenderTargets; ++i) {
		rp.renderTextures.emplace_back();
		auto &rt = rp.renderTextures.back();
		rt.init(texInitData, Dar::FRAME_COUNT);
		rt.setName(L"RenderTexture[" + std::to_wstring(renderPasses.size() - 1) + L"]");
	}

	return true;
}

bool ShaderToy::addRenderPass(const char *code, const String &displayName, int numRenderTargets) {
	auto shaderName = L"shader" + std::to_wstring(shaderCount++);
	bool res = Dar::ShaderCompiler::compileFromSource(code, shaderName, L".\\res\\shaders", Dar::ShaderType::Pixel);

	if (!res) {
		return false;
	}

	return addRenderPass(shaderName, displayName, numRenderTargets);
}

bool ShaderToy::addRenderPass(const String &name) {
	renderPasses.emplace_back();

	RenderPass *&rpp = renderPasses.back();
	rpp = new RenderPass;

	RenderPass &rp = *rpp;
	rp.name = name;
	rp.shaderName = L"shader" + std::to_wstring(shaderCount-1);
	rp.shaderSource = 
		"#include \"common.hlsli\"\n"
		"\n"
		"float4 main(PSInput IN) : SV_Target {\n"
		"\treturn float4(1.f, 0.f, 0.f, 1.f);\n"
		"}\n";
	rp.textEdit.SetText("");
	rp.compiled = false;

	rp.textEdit.SetText(rp.shaderSource);

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
		visited.insert(id);

		if (renderPasses[id]->dependancies.empty()) {
			renderGraph.push_back(id);
			s.pop();
			continue;
		}

		bool hasUnvisitedDependancy = false;
		for (auto d : renderPasses[id]->dependancies) {
			if (visited.find(d) != visited.end()) {
				continue;
			}

			hasUnvisitedDependancy = true;
			s.push(d);
			break;
		}

		if (!hasUnvisitedDependancy) {
			renderGraph.push_back(id);
			s.pop();
		}
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

	return true;
}

void ShaderToy::RenderPass::addRenderTexture(const ShaderToy &app) {
	renderTextures.emplace_back();
	auto &rt = renderTextures.back();

	Dar::TextureInitData texInitData = {};
	texInitData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
	texInitData.width = app.getWidth();
	texInitData.height = app.getHeight();
	rt.init(texInitData, Dar::FRAME_COUNT);
}

void ShaderToy::RenderPass::addResourceTexture(const String &path) {
	if (auto it = std::find(textureDescs.begin(), textureDescs.end(), path); it != textureDescs.end()) {
		return;
	}

	textureDescs.push_back(path);
	needUpdate = true;
}

bool ShaderToy::RenderPass::uploadTextures(Dar::UploadHandle uploadHandle) {
	if (!needUpdate) {
		return true;
	}

	Dar::ResourceManager &resManager = Dar::getResourceManager();

	SizeType numTextures = textureDescs.size();

	std::for_each(
		textures.begin(),
		textures.end(),
		[](Dar::TextureResource &texResource) {
			texResource.deinit();
		}
	);

	textures.resize(numTextures);

	Vector<ImageData> texData(numTextures);
	Vector<D3D12_RESOURCE_DESC> texDescs(numTextures);
	Vector<Dar::TextureInitData> texInitDatas(numTextures);
	for (int i = 0; i < numTextures; ++i) {
		String &tex = textureDescs[i];
		texData[i] = loadImage(tex);

		wchar_t textureName[32] = L"";
		swprintf(textureName, 32, L"Texture[%d]", i);

		Dar::TextureInitData &texInitData = texInitDatas[i];
		texInitData.width = texData[i].width;
		texInitData.height = texData[i].height;
		texInitData.format = DXGI_FORMAT_R8G8B8A8_UNORM;

		Dar::ResourceInitData resInitData = {};
		resInitData.init(Dar::ResourceType::TextureBuffer);
		resInitData.textureData = texInitData;
		resInitData.name = textureName;

		texDescs[i] = resInitData.getResourceDescriptor();
	}

	resManager.createHeap(texDescs.data(), static_cast<UINT>(texDescs.size()), texturesHeap);

	if (texturesHeap == INVALID_HEAP_HANDLE) {
		return false;
	}

	SizeType heapOffset = 0;
	for (int i = 0; i < numTextures; ++i) {
		Dar::HeapInfo heapInfo = {};
		heapInfo.handle = texturesHeap;
		heapInfo.offset = heapOffset;

		Dar::TextureInitData initData = {};
		initData.width = texData[i].width;
		initData.height = texData[i].height;
		initData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
		textures[i].init(initData, Dar::TextureResourceType::ShaderResource, &heapInfo);

		UINT64 size = textures[i].upload(uploadHandle, texData[i].data);

		heapOffset += size;
	}

	needUpdate = false;
	return true;
}
