#include "d3d12/pipeline_state.h"

#include "math/dar_math.h"

#include "reslib/resource_library.h"

#include <filesystem>

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

	auto sname = desc.shaderName;
	auto &reslib = getResourceLibrary();

	auto psShaderName = sname + "_ps";
	auto psShader = reslib.getShader(psShaderName);
	if (psShader == nullptr) {
		LOG_FMT(Error, "Failed to read %s!", psShaderName.c_str());
		return false;
	}
	stream.insert(PixelShaderToken(D3D12_SHADER_BYTECODE{ psShader->GetBufferPointer(), psShader->GetBufferSize() }));

	if (mask & shaderInfoFlags_useVertex) {
		auto vsShaderName = sname + "_vs";
		if (auto vsShader = reslib.getShader(vsShaderName)) {
			stream.insert(VertexShaderToken({ vsShader->GetBufferPointer(), vsShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", vsShaderName.c_str());
			return false;
		}
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
	}

	if (mask & shaderInfoFlags_useGeometry) {
		auto gsShaderName = sname + "_gs";
		if (auto gsShader = reslib.getShader(gsShaderName)) {
			stream.insert(GeometryShaderToken({ gsShader->GetBufferPointer(), gsShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", gsShaderName.c_str());
			return false;
		}
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
	}

	if (mask & shaderInfoFlags_useDomain) {
		auto dsShaderName = sname + "_ds";
		if (auto dsShader = reslib.getShader(dsShaderName)) {
			stream.insert(DomainShaderToken({ dsShader->GetBufferPointer(), dsShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", dsShaderName.c_str());
			return false;
		}
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS;
	}

	if (mask & shaderInfoFlags_useHull) {
		auto hsShaderName = sname + "_hs";
		if (auto hsShader = reslib.getShader(hsShaderName)) {
			stream.insert(HullShaderToken({ hsShader->GetBufferPointer(), hsShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", hsShaderName.c_str());
			return false;
		}
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
	}

	if (mask & shaderInfoFlags_useMesh) {
		auto msShaderName = sname + "_ms";
		if (auto msShader = reslib.getShader(msShaderName)) {
			stream.insert(MeshShaderToken({ msShader->GetBufferPointer(), msShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", msShaderName.c_str());
			return false;
		}
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS;
	}

	if (mask & shaderInfoFlags_useAmplification) {
		auto asShaderName = sname + "_as";
		if (auto asShader = reslib.getShader(asShaderName)) {
			stream.insert(AmplificationShaderToken({ asShader->GetBufferPointer(), asShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", asShaderName.c_str());
			return false;
		}
	} else {
		rsFlags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS;
	}

	if (mask & shaderInfoFlags_useCompute) {
		auto csShaderName = sname + "_cs";
		if (auto csShader = reslib.getShader(csShaderName)) {
			stream.insert(ComputeShaderToken({ csShader->GetBufferPointer(), csShader->GetBufferSize() }));
		} else {
			LOG_FMT(Error, "Failed to read %s!", csShaderName.c_str());
			return false;
		}
	}

	const int numConstantBufferViews = std::min(UINT(15), desc.numConstantBufferViews);
	const int numParams = numConstantBufferViews;
	Vector<CD3DX12_ROOT_PARAMETER1> rsParams(numParams);
	for (int i = 0; i < numConstantBufferViews; ++i) {
		rsParams[i].InitAsConstantBufferView(i);
	}

	CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
	rootSignatureDesc.Init_1_1(numParams, rsParams.data(), desc.numStaticSamplers, desc.staticSamplerDescs, rsFlags);

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