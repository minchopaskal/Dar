#pragma once

#include "d3d12/includes.h"
#include "utils/defines.h"
#include "d3dx12.h"

namespace Dar {

template <class DataType, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE D3D12Type>
struct alignas(void *) PipelineStateStreamToken {
	PipelineStateStreamToken() {}

	PipelineStateStreamToken(const DataType &data) : token(data), type(D3D12Type) {}

	PipelineStateStreamToken &operator=(DataType &data) {
		token = data;
		return token;
	}

	operator DataType &() {
		return token;
	}

	DataType &operator&() {
		return token;
	}

	const void *getData() const {
		return this;
	}

	UINT64 getUnderlyingSize() const {
		return sizeof(DataType);
	}

private:
	D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;
	DataType token;
};

using StateFlagsToken = PipelineStateStreamToken<D3D12_PIPELINE_STATE_FLAGS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS>;
using NodeMaskToken = PipelineStateStreamToken<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK>;
using RootSignatureToken = PipelineStateStreamToken<ID3D12RootSignature *, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE>;
using InputLayoutToken = PipelineStateStreamToken<D3D12_INPUT_LAYOUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT>;
using IBStripCutValueToken = PipelineStateStreamToken<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE>;
using PrimitiveTopologyToken = PipelineStateStreamToken<D3D12_PRIMITIVE_TOPOLOGY_TYPE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY>;
using VertexShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS>;
using GeometryShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS>;
using StreamOutputDescToken = PipelineStateStreamToken<D3D12_STREAM_OUTPUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT>;
using HullShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS>;
using DomainShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS>;
using PixelShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS>;
using AmplificationShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS>;
using MeshShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS>;
using ComputeShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS>;
using BlendDescToken = PipelineStateStreamToken<D3D12_BLEND_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND>;
using DepthStencilDescToken = PipelineStateStreamToken<CD3DX12_DEPTH_STENCIL_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL>;
using DepthStencilDesc1Token = PipelineStateStreamToken<D3D12_DEPTH_STENCIL_DESC1, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1>;
using DepthStencilFormatToken = PipelineStateStreamToken<DXGI_FORMAT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT>;
using RasterizerDescToken = PipelineStateStreamToken<D3D12_RASTERIZER_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER>;
using RTFormatsToken = PipelineStateStreamToken<D3D12_RT_FORMAT_ARRAY, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS>;
using SampleDescToken = PipelineStateStreamToken<DXGI_SAMPLE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC>;
using SampleMaskToken = PipelineStateStreamToken<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK>;
using CachedPSOToken = PipelineStateStreamToken<D3D12_CACHED_PIPELINE_STATE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO>;
using ViewInstancingToken = PipelineStateStreamToken<D3D12_VIEW_INSTANCING_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING>;

struct PipelineStateStream {
	PipelineStateStream();

	template <typename T>
	void insert(const T &token) {
		SizeType oldSize = data.size();
		data.resize(data.size() + sizeof(T));
		memcpy(data.data() + oldSize, token.getData(), sizeof(T));
	}

	void *getData();
	SizeType getSize() const;

private:
	Vector<UINT8> data;
};

enum ShaderInfoFlags : UINT8 {
	shaderInfoFlags_useGeometry = (1 << 0),
	shaderInfoFlags_useDomain = (1 << 1),
	shaderInfoFlags_useHull = (1 << 2),
	shaderInfoFlags_useCompute = (1 << 3),
	shaderInfoFlags_useMesh = (1 << 4),
	shaderInfoFlags_useAmplification = (1 << 5),
	shaderInfoFlags_useVertex = (1 << 6)
};

constexpr int MAX_RENDER_TARGETS = 8;

struct PipelineStateDesc {
	WString shaderName = L""; ///< Base name of the shader files
	D3D12_INPUT_ELEMENT_DESC *inputLayouts = nullptr; ///< Input layout descriptions. Ignored if nullptr
	D3D12_ROOT_SIGNATURE_FLAGS *rootSignatureFlags = nullptr; ///< Additional flags for the root signature. Ignored if nullptr.
	D3D_ROOT_SIGNATURE_VERSION maxVersion = D3D_ROOT_SIGNATURE_VERSION_1_0; ///< Root signature features version. Used for root signature creation.
	D3D12_STATIC_SAMPLER_DESC *staticSamplerDesc = nullptr; ///< Static sampler description. Used for root signature creation
	DXGI_FORMAT depthStencilBufferFormat = DXGI_FORMAT_UNKNOWN; ///< Format for the depth stencil buffer. Leave unknown if DSB is null.
	DXGI_FORMAT renderTargetFormats[MAX_RENDER_TARGETS] = { DXGI_FORMAT_R8G8B8A8_UNORM };
	D3D12_CULL_MODE cullMode = D3D12_CULL_MODE_BACK;
	UINT numRenderTargets = 1; ///< Number of render targets
	UINT numInputLayouts = 0; ///< Number of input layouts.
	UINT numTextures = 0; ///< Number of textures in the texture descriptor table.
	UINT numConstantBufferViews = 0; ///< Number of constant buffer views. Used for root signature creation.
	UINT8 shadersMask = 0; ///< Mask indicating which types of shaders will be used. Only the fragment shader is ON by default.
};

struct PipelineState {
	PipelineState();

	bool init(const ComPtr<ID3D12Device> &device, PipelineStateStream &pss);
	bool init(const ComPtr<ID3D12Device> &device, const PipelineStateDesc &desc);

	ID3D12PipelineState *getPipelineState() const;
	ID3D12RootSignature *getRootSignature() const;

private:
	bool initPipeline(const ComPtr<ID3D12Device> &device, PipelineStateStream &pss);

private:
	ComPtr<ID3D12PipelineState> pipelineState;
	ComPtr<ID3D12RootSignature> rootSignature;
};

} // namespace Dar