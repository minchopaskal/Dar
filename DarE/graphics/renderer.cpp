#include "renderer.h"

#include "d3d12/includes.h"
#include "framework/app.h"
#include "utils/defines.h"

#include "imgui.h"
#include "imgui/backends/imgui_impl_dx12.h"
#include "imgui/backends/imgui_impl_glfw.h"

namespace Dar {

struct RenderPass {
	/// Initialize the render pass given the description.
	/// Creates the pipeline state (TODO: load PSO from cache)
	/// @param device Device used for the render pass initialization steps.
	/// @param rpd Render pass description.
	/// @param frameCount How many frames are rendered at the same time. Used for determining the size of the SRV and RTV heaps.
	void init(ComPtr<ID3D12Device> device, Backbuffer *backbuffer, const RenderPassDesc &rpd, int frameCount) {
		auto &psoDesc = rpd.psoDesc;
		pipeline.init(device, psoDesc);

		DXGI_SWAP_CHAIN_DESC1 scDesc = {};
		D3D12_RENDER_PASS_RENDER_TARGET_DESC rtDesc = {};
		bool hasDepthStencilBuffer = false;
		bool hasBackbufferAttachment = false;
		for (int i = 0; i < rpd.attachments.size(); ++i) {
			auto &attachment = rpd.attachments[i];
			switch (attachment.getType()) {
			case RenderPassAttachmentType::RenderTarget:
				if (hasBackbufferAttachment && rpd.attachments[i].isBackbuffer()) {
					dassert(false);
					break;
				}

				if (attachment.isBackbuffer()) {
					dassert(backbuffer);
					if (!backbuffer) {
						break;
					}

					this->backbuffer = backbuffer;
					hasBackbufferAttachment = true;
				}

				// TODO: Add support for different begining/ending accesses if needed
				rtDesc = {};
				rtDesc.cpuDescriptor = { NULL }; // We will update that during rendering
				rtDesc.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
				rtDesc.BeginningAccess.Clear.ClearValue.Format = attachment.isBackbuffer() ? backbuffer->getFormat() : attachment.getFormat();
				rtDesc.BeginningAccess.Clear.ClearValue.Color[0] = 0.f;
				rtDesc.BeginningAccess.Clear.ClearValue.Color[1] = 0.f;
				rtDesc.BeginningAccess.Clear.ClearValue.Color[2] = 0.f;
				rtDesc.BeginningAccess.Clear.ClearValue.Color[3] = 0.f;
				rtDesc.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;

				renderTargetDescs.push_back(rtDesc);
				renderTargetAttachments.push_back(attachment);

				break;
			case RenderPassAttachmentType::DepthStencil:
				if (hasDepthStencilBuffer) {
					dassert(false);
					break;
				}

				depthStencilDesc.cpuDescriptor = attachment.getCPUHandle();
				if (rpd.attachments[i].clearDepthBuffer()) {
					depthStencilDesc.DepthBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
					depthStencilDesc.DepthBeginningAccess.Clear.ClearValue.Format = attachment.getFormat();
					depthStencilDesc.DepthBeginningAccess.Clear.ClearValue.DepthStencil = { 1.f, 0 };
				} else {
					depthStencilDesc.DepthBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE;
				}
				depthStencilDesc.DepthEndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
				depthBufferAttachment = attachment;

				hasDepthStencilBuffer = true;
				break;
			}
		}

		dassert(psoDesc.numRenderTargets == renderTargetAttachments.size());
		dassert((psoDesc.depthStencilBufferFormat != DXGI_FORMAT_UNKNOWN) == hasDepthStencilBuffer);

		for (int i = 0; i < FRAME_COUNT; ++i) {
			rtvHeap[i].init(
				device.Get(),
				D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
				static_cast<int>(renderTargetAttachments.size()), /*numDescriptors*/
				true /*shaderVisible*/
			);
		}

		// We do not initialize the srv heap as this is a job
		// for the setup function. We cannot possibly know beforehand how many
		// resources will be bound to the current pass' pipeline.
		// F.e the number textures can be different on each frame.
	}

	void begin(const FrameData &frameData, CommandList &cmdList, int backbufferIndex) {
		cmdList->SetPipelineState(pipeline.getPipelineState());

		// Prepare the rtv heap. Do this each frame in order not to deal with
		// tracking when the RTV texture is changed and stuff like that.
		rtvHeap[backbufferIndex].reset();
		bool hasBackBuffer = false;
		for (int i = 0; i < renderTargetAttachments.size(); ++i) {
			ID3D12Resource *rtRes = nullptr;
			const bool isBackBuffer = renderTargetAttachments[i].isBackbuffer();
			if (isBackBuffer) {
				dassert(backbuffer && !hasBackBuffer); // Only one backbuffer attachment is allowed!
				rtRes = backbuffer->getBufferResource(backbufferIndex);
				hasBackBuffer = true;
			} else {
				rtRes = renderTargetAttachments[i].getBufferResource(backbufferIndex);
			}

			dassert(rtRes != nullptr);

			rtvHeap[backbufferIndex].addRTV(rtRes, nullptr);
			renderTargetDescs[i].cpuDescriptor = rtvHeap[backbufferIndex].getCPUHandle(i);
			if (isBackBuffer) {
				auto b = CD3DX12_RESOURCE_BARRIER::Transition(
					backbuffer->getBufferResource(backbufferIndex),
					D3D12_RESOURCE_STATE_COMMON,
					D3D12_RESOURCE_STATE_RENDER_TARGET
				);
				cmdList->ResourceBarrier(1, &b);
			} else {
				cmdList.transition(
					renderTargetAttachments[i].getResourceHandle(backbufferIndex),
					D3D12_RESOURCE_STATE_RENDER_TARGET
				);
			}
		}

		/*bool hasDepthStencil = depthBufferAttachment.valid();
		cmdList->BeginRenderPass(static_cast<UINT>(renderTargetDescs.size()), renderTargetDescs.data(), hasDepthStencil ? &depthStencilDesc : NULL, D3D12_RENDER_PASS_FLAG_NONE);*/
	}

	void end(CommandList &cmdList) {
		/*cmdList->EndRenderPass();*/
	}

	void deinit() {
		pipeline.deinit();
		for (int i = 0; i < FRAME_COUNT; ++i) {
			rtvHeap[i].deinit();
			srvHeap[i].deinit();
		}
	}

public:
	PipelineState pipeline;
	Vector<RenderPassAttachment> renderTargetAttachments;
	RenderPassAttachment depthBufferAttachment;
	Vector<D3D12_RENDER_PASS_RENDER_TARGET_DESC> renderTargetDescs;
	Backbuffer *backbuffer;
	DescriptorHeap rtvHeap[FRAME_COUNT];
	DescriptorHeap srvHeap[FRAME_COUNT];
	D3D12_RENDER_PASS_DEPTH_STENCIL_DESC depthStencilDesc;
};

void getHardwareAdapter(
	IDXGIFactory1 *pFactory,
	IDXGIAdapter1 **ppAdapter,
	bool requestHighPerformanceAdapter
) {
	*ppAdapter = nullptr;

	ComPtr<IDXGIAdapter1> adapter;

	ComPtr<IDXGIFactory6> factory6;
	if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
		for (
			UINT adapterIndex = 0;
			DXGI_ERROR_NOT_FOUND != factory6->EnumAdapterByGpuPreference(
				adapterIndex,
				DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
				IID_PPV_ARGS(ppAdapter)
			);
			++adapterIndex) {
			if (adapter == nullptr) {
				continue;
			}

			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	} else {
		for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter); ++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	}

	*ppAdapter = adapter.Detach();
}

bool checkTearingSupport() {
	ComPtr<IDXGIFactory5> dxgiFactory;
	if (!SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)))) {
		return false;
	}

	bool allowTearing = false;
	if (FAILED(dxgiFactory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
		allowTearing = false;
	}

	return (allowTearing == true);
}

Renderer::Renderer() : commandQueueDirect(D3D12_COMMAND_LIST_TYPE_DIRECT), renderPassesStorage(nullptr) {}

void Renderer::addRenderPass(const RenderPassDesc &rpd) {
	renderPassesDesc.push_back(rpd);
}

void Renderer::compilePipeline() {
	renderPasses.clear();
	if (renderPassesStorage != nullptr) {
		delete[] renderPassesStorage;
	}

	const SizeType numRenderPasses = renderPassesDesc.size();
	renderPassesStorage = new Byte[numRenderPasses * sizeof(RenderPass)];
	for (SizeType i = 0; i < numRenderPasses; ++i) {
		RenderPass *renderPass = new (renderPassesStorage + i * sizeof(RenderPass)) RenderPass;
		renderPass->init(device, &backbuffer, renderPassesDesc[i], FRAME_COUNT);

		renderPasses.push_back(renderPass);
	}

	// We no longer need the memory since the render passes are initialized.
	renderPassesDesc.clear();
}

void Renderer::init() {
	App *app = getApp();
	viewport = { 0.f, 0.f, static_cast<float>(app->getWidth()), static_cast<float>(app->getHeight()), 0.f, 1.f };

	initDevice();

	initImGui();
}

void Renderer::deinit() {
	for (int i = 0; i < renderPasses.size(); ++i) {
		renderPasses[i]->deinit();
	}
	renderPasses.clear();
	if (renderPassesStorage) {
		delete renderPassesStorage;
		renderPassesStorage = nullptr;
	}

	flush();
	commandQueueDirect.deinit();

	backbuffer.deinit();
	auto refCnt = device.Reset();

#ifdef DAR_DEBUG
	HMODULE dxgiModule = GetModuleHandleA("Dxgidebug.dll");
	if (!dxgiModule) {
		return;
	}

	using DXGIGetDebugInterfaceProc = HRESULT (*)(REFIID riid, void **ppDebug);

	auto dxgiGetDebugInterface = (DXGIGetDebugInterfaceProc)GetProcAddress(dxgiModule, "DXGIGetDebugInterface");

	if (!dxgiGetDebugInterface) {
		return;
	}

	ComPtr<IDXGIDebug> debugLayer;
	if (SUCCEEDED(dxgiGetDebugInterface(IID_PPV_ARGS(&debugLayer)))) {
		debugLayer->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_IGNORE_INTERNAL | DXGI_DEBUG_RLO_SUMMARY));
	}
#endif // DAR_DEBUG
}

bool Renderer::deinitImGui() {
	ImGui_ImplDX12_InvalidateDeviceObjects();
	if (!imGuiShutdown) {
		ImGui_ImplDX12_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		imGuiShutdown = true;
	}
	imguiSRVHeap.Reset();

	return true;
}

void Renderer::flush() {
	commandQueueDirect.flush();
}

void Renderer::beginFrame() {}

void Renderer::renderFrame(const FrameData &frameData) {
	commandQueueDirect.addCommandListForExecution(populateCommandList(frameData));
	fenceValues[backbufferIndex] = commandQueueDirect.executeCommandLists();

	++numRenderedFrames;

	const UINT syncInterval = settings.vSyncEnabled ? 1 : 0;
	const UINT presentFlags = allowTearing && !settings.vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(backbuffer.present(syncInterval, presentFlags), , "Failed to execute command list!");

	backbufferIndex = backbuffer.getCurrentBackBufferIndex();

	// wait for the next frame's buffer
	commandQueueDirect.waitForFenceValue(fenceValues[backbufferIndex]);
}

void Renderer::endFrame() {}

bool Renderer::registerBackBuffersInResourceManager() {
	return backbuffer.registerInResourceManager();
}

bool Renderer::resizeBackBuffers() {
	App *app = getApp();

	viewport.Width = app->getWidth();
	viewport.Height = app->getHeight();

	if (!backbuffer.resize()) {
		return false;
	}
	
	backbufferIndex = backbuffer.getCurrentBackBufferIndex();

	deinitImGui();
	initImGui();

	return true;
}

bool Renderer::initDevice() {
#if defined(DAR_DEBUG)
	/* Enable debug layer */
	{
		ComPtr<ID3D12Debug> debugLayer;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugLayer)))) {
			debugLayer->EnableDebugLayer();
		}
	}
#endif // defined(DAR_DEBUG)

	/* Create the device */
	ComPtr<IDXGIFactory4> dxgiFactory;
	UINT createFactoryFlags = 0;
#if defined(DAR_DEBUG)
	createFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif // defined(DAR_DEBUG)

	RETURN_FALSE_ON_ERROR(
		CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory)),
		"Error while creating DXGI Factory!"
	);

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	getHardwareAdapter(dxgiFactory.Get(), &hardwareAdapter, false);

	RETURN_FALSE_ON_ERROR(D3D12CreateDevice(
		hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)),
		"Failed to create device!"
	);

	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_5 };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
		"Device does not support shader model 6.5!"
	);

	if (shaderModel.HighestShaderModel != D3D_SHADER_MODEL_6_5) {
		fprintf(stderr, "Shader model 6.5 not supported!");
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS options = { };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)),
		"Failed to check features support!"
	);

	if (options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
		fprintf(stderr, "GPU does not support resource binding tier 3!");
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = { };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7)),
		"Failed to check features support!"
	);

	// Note: we don't do mesh shading currently, yet we will, so we want to support it.
	/*if (options7.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED) {
		fprintf(stderr, "Mesh shading is not supported!");
		return false;
	}*/

	/* Create command queue */
	commandQueueDirect.init(device);

	/* Create a swap chain */
	allowTearing = checkTearingSupport();

	backbuffer.init(dxgiFactory, commandQueueDirect, allowTearing);
	backbufferIndex = backbuffer.getCurrentBackBufferIndex();

	return true;
}

bool Renderer::initImGui() {
	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = 1;
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&imguiSRVHeap)),
		"Failed to create ImGui's SRV descriptor heap!"
	);

	auto *app = getApp();

	ImGuiIO &io = ImGui::GetIO();
	io.DisplaySize.x = float(app->getWidth());
	io.DisplaySize.y = float(app->getHeight());

	ImGui_ImplGlfw_InitForOther(app->getGLFWWindow(), true);
	ImGui_ImplDX12_Init(
		device.Get(),
		FRAME_COUNT,
		DXGI_FORMAT_R8G8B8A8_UNORM,
		imguiSRVHeap.Get(),
		imguiSRVHeap->GetCPUDescriptorHandleForHeapStart(),
		imguiSRVHeap->GetGPUDescriptorHandleForHeapStart()
	);

	ImGui_ImplDX12_CreateDeviceObjects();

	imGuiShutdown = false;

	return true;
}

void Renderer::renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle) {
	if (!settings.useImGui) {
		return;
	}

	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	getApp()->drawUI();

	ImGui::Render();

	cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
	cmdList->SetDescriptorHeaps(1, imguiSRVHeap.GetAddressOf());

	ComPtr<ID3D12GraphicsCommandList> gCmdList;
	cmdList.getComPtr().As(&gCmdList);
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), gCmdList.Get());
}

CommandList Renderer::populateCommandList(const FrameData &frameData) {
	CommandList cmdList = commandQueueDirect.getCommandList();

	cmdList->RSSetViewports(1, &viewport);
	cmdList->RSSetScissorRects(1, &scissorRect);

	// Cache the rtv handle. The last rtv handle will be used for the ImGui rendering.
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = {};
	for (int renderPassIndex = 0; renderPassIndex < renderPasses.size(); ++renderPassIndex) {
		RenderPass &renderPass = *renderPasses[renderPassIndex];
		renderPass.begin(frameData, cmdList, backbufferIndex);

		auto &shaderResources = frameData.shaderResources[renderPassIndex];
		const int numShaderResources = shaderResources.size();
		auto &srvHeap = renderPass.srvHeap[backbufferIndex];
		if (numShaderResources > 0 && (!srvHeap || srvHeap.getNumViews() < numShaderResources)) {
			srvHeap.init(
				device.Get(),
				D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
				numShaderResources,
				true
			);
		}

		srvHeap.reset();
		for (auto &res : shaderResources) {
			ResourceHandle handle = {};
			bool isTex = false;
			switch (res.type) {
			case FrameData::ShaderResourceType::Data:
				handle = res.data->getHandle();
				break;
			case FrameData::ShaderResourceType::Texture:
				isTex = true;
				handle = res.tex->getHandle();
				break;
			default:
				dassert(false);
				break;
			}
			cmdList.transition(handle, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			if (isTex) {
				srvHeap.addTexture2DSRV(handle.get(), res.tex->getFormat());
			} else {
				srvHeap.addBufferSRV(handle.get(), res.data->getNumElements(), res.data->getElementSize());
			}
		}

		if (renderPass.srvHeap[backbufferIndex]) {
			cmdList->SetDescriptorHeaps(1, renderPass.srvHeap[backbufferIndex].getAddressOf());
		}

		cmdList->SetGraphicsRootSignature(renderPass.pipeline.getRootSignature());

		for (auto &cb : frameData.constantBuffers) {
			cmdList.transition(cb.handle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
			cmdList.setConstantBufferView(cb.rootParameterIndex, cb.handle);
		}

		const bool hasDepthBuffer = renderPass.depthBufferAttachment.valid();
		const int numRenderTargets = static_cast<int>(renderPass.renderTargetDescs.size());
		rtvHandle = renderPass.rtvHeap[backbufferIndex].getCPUHandle(0);
		const D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = hasDepthBuffer ? renderPass.depthBufferAttachment.getCPUHandle() : D3D12_CPU_DESCRIPTOR_HANDLE{ 0 };

		constexpr float clearColor[] = { 0.f, 0.f, 0.f, 0.f };
		for (int i = 0; i < numRenderTargets; ++i) {
			cmdList->ClearRenderTargetView(renderPass.rtvHeap[backbufferIndex].getCPUHandle(i), clearColor, 0, nullptr);
		}

		if (hasDepthBuffer && renderPass.depthBufferAttachment.clearDepthBuffer()) {
			cmdList.transition(renderPass.depthBufferAttachment.getResourceHandle(0), D3D12_RESOURCE_STATE_DEPTH_WRITE);
			cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);
		}

		cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		if (frameData.vertexBuffer) {
			cmdList.transition(frameData.vertexBuffer->bufferHandle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
			cmdList->IASetVertexBuffers(0, 1, &frameData.vertexBuffer->bufferView);
		}
		if (frameData.indexBuffer) {
			cmdList.transition(frameData.indexBuffer->bufferHandle, D3D12_RESOURCE_STATE_INDEX_BUFFER);
			cmdList->IASetIndexBuffer(&frameData.indexBuffer->bufferView);
		}

		cmdList->OMSetRenderTargets(numRenderTargets, &rtvHandle, TRUE, hasDepthBuffer ? &dsvHandle : nullptr);

		frameData.renderCommands[renderPassIndex].execCommands(cmdList);

		renderPass.end(cmdList);
	}

	renderUI(cmdList, rtvHandle);
	cmdList.transition(backbuffer.getHandle(backbufferIndex), D3D12_RESOURCE_STATE_PRESENT);

	return cmdList;
}

void RenderPassDesc::attach(const RenderPassAttachment &rpa) {
	attachments.push_back(rpa);
}

void RenderTarget::init(TextureInitData &texInitData, UINT numFramesInFlight) {
	dassert(numFramesInFlight <= FRAME_COUNT);
	numFramesInFlight = dmath::min(numFramesInFlight, FRAME_COUNT);
	this->numFramesInFlight = numFramesInFlight;

	for (UINT i = 0; i < numFramesInFlight; ++i) {
		rtTextures[i].init(texInitData, TextureResourceType::RenderTarget);
	}
}

ResourceHandle RenderTarget::getHandleForFrame(UINT frameIndex) const {
	dassert(frameIndex < numFramesInFlight);
	if (frameIndex >= numFramesInFlight) {
		return INVALID_RESOURCE_HANDLE;
	}

	return rtTextures[frameIndex].getHandle();
}

ID3D12Resource *RenderTarget::getBufferResourceForFrame(UINT frameIndex) const {
	dassert(frameIndex < numFramesInFlight);
	if (frameIndex >= numFramesInFlight) {
		return nullptr;
	}

	return rtTextures[frameIndex].getBufferResource();
}

void RenderTarget::resizeRenderTarget(int width, int height) {
	Dar::TextureInitData rtvTextureDesc = {};
	rtvTextureDesc.width = width;
	rtvTextureDesc.height = height;
	rtvTextureDesc.format = getFormat();
	rtvTextureDesc.clearValue.color[0] = 0.f;
	rtvTextureDesc.clearValue.color[1] = 0.f;
	rtvTextureDesc.clearValue.color[2] = 0.f;
	rtvTextureDesc.clearValue.color[3] = 0.f;

	auto &resManager = getResourceManager();
	for (UINT i = 0; i < numFramesInFlight; ++i) {
		TextureResource &t = rtTextures[i];
		const WString &name = t.getName();

		t.init(rtvTextureDesc, TextureResourceType::RenderTarget);
		t.setName(name);
	}
}

void RenderTarget::setName(const WString &name) {
	for (UINT i = 0; i < numFramesInFlight; ++i) {
		rtTextures[i].setName(name + L"[" + std::to_wstring(i) + L"]");
	}
}

bool Backbuffer::init(ComPtr<IDXGIFactory4> dxgiFactory, CommandQueue &commandQueue, bool allowTearing) {
	App *app = getApp();

	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	scDesc.Width = app->getWidth();
	scDesc.Height = app->getHeight();
	scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scDesc.Stereo = FALSE;
	// NOTE: if multisampling should be enabled the bitblt transfer swap method should be used
	scDesc.SampleDesc = { 1, 0 }; // Not using multi-sampling.
	scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scDesc.BufferCount = FRAME_COUNT;
	scDesc.Scaling = DXGI_SCALING_STRETCH;
	scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	scDesc.Flags = allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

	ComPtr<IDXGISwapChain1> swapChainPlaceholder;
	RETURN_FALSE_ON_ERROR(
		dxgiFactory->CreateSwapChainForHwnd(
			commandQueue.getCommandQueue().Get(), // Swap chain needs the queue so that it can force a flush on it.
			app->getWindow(),
			&scDesc,
			NULL /*DXGI_SWAP_CHAIN_FULLSCREEN_DESC*/, // will be handled manually
			NULL /*pRestrictToOutput*/,
			&swapChainPlaceholder
		),
		"Failed to create swap chain!"
	);

	RETURN_FALSE_ON_ERROR(
		dxgiFactory->MakeWindowAssociation(app->getWindow(), DXGI_MWA_NO_ALT_ENTER),
		"Failed to make window association NO_ALT_ENTER"
	);

	RETURN_FALSE_ON_ERROR(
		swapChainPlaceholder.As(&swapChain),
		"Failed to create swap chain!"
	);

	for (UINT i = 0; i < FRAME_COUNT; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);

		wchar_t backBufferName[32];
		swprintf(backBufferName, 32, L"BackBuffer[%u]", i);
		backBuffers[i]->SetName(backBufferName);
	}

	return true;
}

bool Backbuffer::registerInResourceManager() {
	auto &resManager = getResourceManager();

	for (UINT i = 0; i < FRAME_COUNT; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);

		// Register the back buffer's resources manually since the resource manager doesn't own them, the swap chain does.
#ifdef DAR_DEBUG
		backBuffersHandles[i] = resManager.registerResource(
			backBuffers[i].Get(),
			1, /*numSubresource*/
			0, /*size*/ // we don't use that
			D3D12_RESOURCE_STATE_COMMON, // initial state
			ResourceType::RenderTargetBuffer
		);
#else
		backBuffersHandles[i] = resManager.registerResource(backBuffers[i].Get(), 1, 0, D3D12_RESOURCE_STATE_COMMON);
#endif

		wchar_t backBufferName[32];
		swprintf(backBufferName, 32, L"BackBuffer[%u]", i);
		backBuffers[i]->SetName(backBufferName);
	}

	return true;
}

bool Backbuffer::resize() {
	auto &resManager = getResourceManager();

	App *app = getApp();
	int width = app->getWidth();
	int height = app->getHeight();

	for (unsigned int i = 0; i < FRAME_COUNT; ++i) {
		backBuffers[i].Reset();
		// It's important to deregister an outside resource if you want it deallocated
		// since the ResourceManager keeps a ref if it was registered with it.
		resManager.deregisterResource(backBuffersHandles[i]);
	}

	DXGI_SWAP_CHAIN_DESC scDesc = { };
	RETURN_FALSE_ON_ERROR(
		swapChain->GetDesc(&scDesc),
		"Failed to retrieve swap chain's description"
	);
	RETURN_FALSE_ON_ERROR(
		swapChain->ResizeBuffers(
			FRAME_COUNT,
			width,
			height,
			scDesc.BufferDesc.Format,
			scDesc.Flags
		),
		"Failed to resize swap chain buffer"
	);

	return registerInResourceManager();
}

UINT Backbuffer::getCurrentBackBufferIndex() const {
	return swapChain->GetCurrentBackBufferIndex();
}

DXGI_FORMAT Backbuffer::getFormat() const {
	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	RETURN_ON_ERROR(swapChain->GetDesc1(&scDesc), DXGI_FORMAT_UNKNOWN, "Failed to retrieve swap chain's description");

	return scDesc.Format;
}

HRESULT Backbuffer::present(UINT syncInterval, UINT flags) const {
	return swapChain->Present(syncInterval, flags);
}

void Backbuffer::deinit() {
	swapChain.Reset();
	for (int i = 0; i < FRAME_COUNT; ++i) {
		backBuffers[i].Reset();
	}
}

void FrameData::beginFrame(const Renderer &renderer) {
	vertexBuffer = nullptr;
	indexBuffer = nullptr;
	constantBuffers.clear();
	shaderResources.clear();
	shaderResources.resize(renderer.getNumPasses());
	
	if (!useSameCommands) {
		renderCommands.clear();
		renderCommands.resize(renderer.getNumPasses());
	}
}

RenderCommandList::~RenderCommandList() {
	delete[] memory;
	memory = nullptr;
	size = 0;
}

void RenderCommandList::execCommands(CommandList &cmdList) const {
	using RenderCommandIterator = Byte*;
	RenderCommandIterator it = memory;
	while (it != memory + size) {
		auto *renderCommand = std::bit_cast<RenderCommandInvalid*>(it);
		SizeType rcSize = 0;
		switch (renderCommand->type) {
		using enum RenderCommandType;
		case DrawInstanced:
			std::bit_cast<RenderCommandDrawInstanced*>(it)->exec(cmdList);
			rcSize = sizeof(RenderCommandDrawInstanced);
			break;
		case DrawIndexedInstanced:
			std::bit_cast<RenderCommandDrawIndexedInstanced *>(it)->exec(cmdList);
			rcSize = sizeof(RenderCommandDrawIndexedInstanced);
			break;
		case SetConstantBuffer:
			std::bit_cast<RenderCommandSetConstantBuffer *>(it)->exec(cmdList);
			rcSize = sizeof(RenderCommandSetConstantBuffer);
			break;
		case Transition:
			std::bit_cast<RenderCommandTransition *>(it)->exec(cmdList);
			rcSize = sizeof(RenderCommandTransition);
			break;
		default:
			dassert(false);
			break;
		}
		it += rcSize;
	}
}

} // namespace Dar
