#include "d3d12/pipeline_state.h"

#include "asset_manager/asset_manager.h"
#include "math/dar_math.h"

#include "d3dcompiler.h"

namespace Dar {

PipelineStateStream::PipelineStateStream() {}

void *PipelineStateStream::getData() {
	return data.data();
}

SizeType PipelineStateStream::getSize() const {
	return data.size();
}

PipelineState::PipelineState() {
	pipelineState.Reset();
	rootSignature.Reset();
}

struct D3D12Empty {};

bool PipelineState::init(const ComPtr<ID3D12Device> &device, PipelineStateStream &pss) {
	if (!initPipeline(device, pss)) {
		return false;
	}

	using EmptyToken = PipelineStateStreamToken<D3D12Empty, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MAX_VALID>;

	UINT8 *pipelineData = reinterpret_cast<UINT8 *>(pss.getData());
	while (pipelineData != pipelineData + pss.getSize()) {
		EmptyToken *emptyToken = reinterpret_cast<EmptyToken *>(pipelineData); // some template magick

		D3D12_PIPELINE_STATE_SUBOBJECT_TYPE *type = reinterpret_cast<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE *>(pipelineData);
		if (*type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) {
			pipelineData += sizeof(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE);
			this->rootSignature = ComPtr<ID3D12RootSignature>{ reinterpret_cast<ID3D12RootSignature *>(pipelineData) };
			break;
		}

		pipelineData += emptyToken->getUnderlyingSize();
	}

	return true;
}

bool PipelineState::init(const ComPtr<ID3D12Device> &device, const PipelineStateDesc &desc) {
	PipelineStateStream stream;

	auto mask = desc.shadersMask;
	auto &base = desc.shaderName;
	auto *rootSignatureFlags = desc.rootSignatureFlags;
	D3D12_ROOT_SIGNATURE_FLAGS rsFlags =
		D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
		D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;

	if (mask & shaderInfoFlags_useVertex) {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
	}
	if (rootSignatureFlags != nullptr) {
		rsFlags |= *rootSignatureFlags;
	}

	ComPtr<ID3DBlob> psShader;
	RETURN_FALSE_ON_ERROR(
		D3DReadFileToBlob(
			getAssetFullPath((base + L"_ps.bin").c_str(), AssetType::Shader).c_str(),
			&psShader
		),
		"Failed to read pixel shader!"
	);
	stream.insert(PixelShaderToken(D3D12_SHADER_BYTECODE{ psShader->GetBufferPointer(), psShader->GetBufferSize() }));

	ComPtr<ID3DBlob> vsShader;
	if (mask & shaderInfoFlags_useVertex) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_vs.bin").c_str(), AssetType::Shader).c_str(),
				&vsShader
			),
			"Failed to read vertex shader!"
		);
		stream.insert(VertexShaderToken({ vsShader->GetBufferPointer(), vsShader->GetBufferSize() }));
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
	}

	ComPtr<ID3DBlob> geomShader;
	if (mask & shaderInfoFlags_useGeometry) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_gs.bin").c_str(), AssetType::Shader).c_str(),
				&geomShader
			),
			"Failed to read geometry shader!"
		);
		stream.insert(GeometryShaderToken({ geomShader->GetBufferPointer(), geomShader->GetBufferSize() }));
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
	}

	ComPtr<ID3DBlob> domShader;
	if (mask & shaderInfoFlags_useDomain) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_ds.bin").c_str(), AssetType::Shader).c_str(),
				&domShader
			),
			"Failed to read domain shader!"
		);
		stream.insert(DomainShaderToken({ domShader->GetBufferPointer(), domShader->GetBufferSize() }));
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS;
	}

	ComPtr<ID3DBlob> hullShader;
	if (mask & shaderInfoFlags_useHull) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_hs.bin").c_str(), AssetType::Shader).c_str(),
				&hullShader
			),
			"Failed to read hull shader!"
		);
		stream.insert(HullShaderToken({ hullShader->GetBufferPointer(), hullShader->GetBufferSize() }));
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
	}

	ComPtr<ID3DBlob> compShader;
	if (mask & shaderInfoFlags_useCompute) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_cs.bin").c_str(), AssetType::Shader).c_str(),
				&compShader
			),
			"Failed to read compute shader!"
		);
		stream.insert(ComputeShaderToken({ compShader->GetBufferPointer(), compShader->GetBufferSize() }));
	}

	ComPtr<ID3DBlob> meshShader;
	if (mask & shaderInfoFlags_useMesh) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_ms.bin").c_str(), AssetType::Shader).c_str(),
				&meshShader
			),
			"Failed to read mesh shader!"
		);
		stream.insert(MeshShaderToken({ meshShader->GetBufferPointer(), meshShader->GetBufferSize() }));
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS;
	}

	ComPtr<ID3DBlob> ampShader;
	if (mask & shaderInfoFlags_useAmplification) {
		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath((base + L"_gs.bin").c_str(), AssetType::Shader).c_str(),
				&ampShader
			),
			"Failed to read amplification shader!"
		);
		stream.insert(AmplificationShaderToken({ ampShader->GetBufferPointer(), ampShader->GetBufferSize() }));
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS;
	}

	const int numConstantBufferViews = dmath::min(UINT(15), desc.numConstantBufferViews);
	const int numParams = numConstantBufferViews + (desc.numTextures > 0 ? 1 : 0);
	Vector<CD3DX12_ROOT_PARAMETER1> rsParams(numParams);
	for (int i = 0; i < numConstantBufferViews; ++i) {
		rsParams[i].InitAsConstantBufferView(i);
	}

	CD3DX12_DESCRIPTOR_RANGE1 range;
	range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, desc.numTextures, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
	if (desc.numTextures > 0) {
		rsParams[numConstantBufferViews].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);
	}

	CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
	rootSignatureDesc.Init_1_1(numParams, rsParams.data(), desc.staticSamplerDesc ? 1 : 0, desc.staticSamplerDesc, rsFlags);

	D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignatureFeatureData = {};
	// cache root signature's feature version
	rootSignatureFeatureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
	if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSignatureFeatureData, sizeof(rootSignatureFeatureData)))) {
		rootSignatureFeatureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}

	ComPtr<ID3DBlob> signature;
	ComPtr<ID3DBlob> error;
	RETURN_FALSE_ON_ERROR(
		D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, rootSignatureFeatureData.HighestVersion, signature.GetAddressOf(), error.GetAddressOf()),
		"Failed to create root signature!"
	);

	RETURN_FALSE_ON_ERROR(
		device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(rootSignature.GetAddressOf())),
		"Failed to create root signature!"
	);

	stream.insert(RootSignatureToken{ rootSignature.Get() });

	stream.insert(PrimitiveTopologyToken{ D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE });

	D3D12_RT_FORMAT_ARRAY rtFormat = {};
	rtFormat.NumRenderTargets = desc.numRenderTargets;
	for (UINT i = 0; i < rtFormat.NumRenderTargets; ++i) {
		rtFormat.RTFormats[i] = desc.renderTargetFormats[i];
	}
	stream.insert(RTFormatsToken{ rtFormat });

	stream.insert(DepthStencilFormatToken{ desc.depthStencilBufferFormat });

	D3D12_RASTERIZER_DESC rd = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	rd.CullMode = desc.cullMode;
	stream.insert(RasterizerDescToken{ rd });

	if (desc.inputLayouts) {
		stream.insert(InputLayoutToken(D3D12_INPUT_LAYOUT_DESC{ desc.inputLayouts, desc.numInputLayouts }));
	}

	return initPipeline(device, stream);
}

ID3D12PipelineState *PipelineState::getPipelineState() const {
	return pipelineState.Get();
}

ID3D12RootSignature *PipelineState::getRootSignature() const {
	return rootSignature.Get();
}

void PipelineState::deinit() {
	pipelineState.Reset();
	rootSignature.Reset();
}

bool PipelineState::initPipeline(const ComPtr<ID3D12Device> &device, PipelineStateStream &pss) {
	D3D12_PIPELINE_STATE_STREAM_DESC pipelineDesc = {};
	pipelineDesc.pPipelineStateSubobjectStream = pss.getData();
	pipelineDesc.SizeInBytes = pss.getSize();

	ComPtr<ID3D12Device2> device2;
	RETURN_FALSE_ON_ERROR(device.As(&device2), "Failed to aquire ID3D12Device2 interface!");

	RETURN_FALSE_ON_ERROR(
		device2->CreatePipelineState(&pipelineDesc, IID_PPV_ARGS(pipelineState.GetAddressOf())),
		"Failed to create pipeline state!"
	);

	return true;
}

} // namespace Dar