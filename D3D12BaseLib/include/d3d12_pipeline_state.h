#pragma once

#include "d3d12_includes.h"
#include "d3d12_defines.h"
#include "d3dx12.h"

template <class DataType, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE D3D12Type>
struct alignas(void*) PipelineStateStreamToken {
	PipelineStateStreamToken() { }

	PipelineStateStreamToken(const DataType &data) {
		token = data;
		type = D3D12Type;
	}

	PipelineStateStreamToken& operator=(DataType &data) {
		token = data;
		return token;
	}

	operator DataType&() {
		return token;
	}

	DataType& operator&() {
		return token;
	}

	void* getData() {
		return this;
	}

private:
	D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;
	DataType token;
};

using StateFlagsToken = PipelineStateStreamToken<D3D12_PIPELINE_STATE_FLAGS, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS>;
using NodeMaskToken = PipelineStateStreamToken<UINT, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK>;
using RootSignatureToken = PipelineStateStreamToken<ID3D12RootSignature*, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE>;
using InputLayoutToken = PipelineStateStreamToken<D3D12_INPUT_LAYOUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT>;
using IBStripCutValueToken = PipelineStateStreamToken<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE>;
using PrimitiveTopologyToken = PipelineStateStreamToken<D3D12_PRIMITIVE_TOPOLOGY_TYPE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY>;
using VertexShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS>;
using GeometryShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS>;
using StreamOutputDescToken = PipelineStateStreamToken<D3D12_STREAM_OUTPUT_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT>;
using HullShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS>;
using DomainShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS>;
using PixelShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS>;
using AssemblyShaderToken = PipelineStateStreamToken<D3D12_SHADER_BYTECODE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS>;
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
	void insert(T &token) {
		size_t oldSize = data.size();
		data.resize(data.size() + sizeof(T));
		memcpy(data.data() + oldSize, token.getData(), sizeof(T));
	}

	void* getData();
	size_t getSize();

private:
	Vector<UINT8> data;
};