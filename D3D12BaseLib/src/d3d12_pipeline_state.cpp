#include "..\include\d3d12_pipeline_state.h"

PipelineStateStream::PipelineStateStream() {
}

void* PipelineStateStream::getData() {
	return data.data();
}

size_t PipelineStateStream::getSize() {
	return data.size();
}
