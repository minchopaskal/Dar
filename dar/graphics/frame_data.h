#pragma once

#include "d3d12/vertex_index_buffer.h"
#include "d3d12/data_buffer.h"
#include "renderer.h"
#include "render_command_list.h"

namespace Dar {

struct FrameData {
	friend class Renderer;
	friend struct RenderPass;

	void clear();

	void beginFrame(const Renderer &renderer);

	void endFrame(const Renderer &) {}

	void startNewPass();

	void setVertexBuffer(VertexBuffer *vb) {
		vertexBuffer = vb;
	}

	void setIndexBuffer(IndexBuffer *ib) {
		indexBuffer = ib;
	}

	/// Add common constant buffer that will be bound to all passes of the pipeline.
	/// @note Any render commands binding constant buffers will be executed after
	/// all common constant buffers are bound.
	void addConstResource(const ResourceHandle &handle, int rootParamterIndex) {
		constantBuffers.emplace_back(handle, rootParamterIndex);
	}

	void addDataBufferResource(const DataBufferResource &dataBuf) {
		ShaderResource shaderResource = {};
		shaderResource.data = &dataBuf;
		shaderResource.type = ShaderResourceType::Data;
		shaderResources[passIndex].push_back(shaderResource);
	}

	void addTextureResource(const TextureResource &texRes) {
		ShaderResource shaderResource = {};
		shaderResource.tex = &texRes;
		shaderResource.type = ShaderResourceType::Texture;
		shaderResources[passIndex].emplace_back(shaderResource);
	}

	void addTextureCubeResource(const TextureResource &texRes) {
		ShaderResource shaderResource = {};
		shaderResource.tex = &texRes;
		shaderResource.type = ShaderResourceType::TextureCube;
		shaderResources[passIndex].emplace_back(shaderResource);
	}
	
	void addRenderCommand(RenderCommandConcept auto renderCmd) {
		if (!useSameCommands) {
			renderCommands[passIndex].addRenderCommand(renderCmd);
		}
	}

	void addUploadContextToWait(UploadContextHandle handle) {
		uploadsToWait.push_back(handle);
	}

	/// Optimization. If set to true doesn't update
	// the commands on the next frame.
	void setUseSameCommands(bool use) {
		useSameCommands = use;
	}

	bool isPassEmpty(int passIdx) const {
		return renderCommands[passIdx].empty();
	}

private:
	struct ConstantResource {
		ResourceHandle handle;
		int rootParameterIndex;
	};

	enum class ShaderResourceType {
		Data,
		Texture,
		TextureCube,
	};

	struct ShaderResource {
		union {
			const TextureResource *tex;
			const DataBufferResource *data;
		};
		ShaderResourceType type;
	};

	/// Constant buffers are bind to all passes of the pipeline.
	Vector<ConstantResource> constantBuffers;

	/// Shader resources for each pass of the pipeline.
	/// Mapping is 1-1, so for pass 0 the renderer would look
	/// its resources in shaderResources[0].
	Vector<Vector<ShaderResource>> shaderResources;

	/// List of render commands for each pass. The list is executed as is,
	/// i.e it preserves the order of the commands as they were passed.
	Vector<RenderCommandList> renderCommands;

	Vector<UploadContextHandle> uploadsToWait;

	VertexBuffer *vertexBuffer = nullptr;
	IndexBuffer *indexBuffer = nullptr;

	int passIndex = -1;

	bool useSameCommands = false;
};

} // namespace Dar
