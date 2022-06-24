#include "shader_compiler.h"

#include <dxcapi.h>

namespace Dar {

WString shaderTypeToStr(ShaderType type) {
	switch (type) {
	case ShaderType::Vertex:
		return L"vs";
	case ShaderType::Pixel:
		return L"ps";
	default:
		// IMPLEMENT ME
		break;
	}

	return L"";
}

ShaderCompilerResult ShaderCompiler::compileFromFile(const WString &filename) {
	return {};
}

ShaderCompilerResult ShaderCompiler::compileFromSource(const char *src, const WString &name, const WString &outputDir, ShaderType type) {
	ComPtr<IDxcLibrary> library;
	if (!SUCCEEDED(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library)))) {
		return {};
	}

	ComPtr<IDxcCompiler2> compiler;
	if (!SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)))) {
		return {};
	}

	uint32_t codePage = CP_UTF8;
	ComPtr<IDxcBlobEncoding> sourceBlob;
	library->CreateBlobWithEncodingOnHeapCopy(src, strlen(src), 0, &sourceBlob);

	WString entryPoint = L"main";
	WString target = shaderTypeToStr(type) + L"_6_6";
	
	constexpr int NUM_ARGUMENTS = 2;
	WString args[NUM_ARGUMENTS];

	args[0] = (L"-Fo " + outputDir + L"\\" + name + L"_" + shaderTypeToStr(type) + L".bin");
	args[1] = L"-I .\\res\\shaders\\";

	const wchar_t *argsPtrs[NUM_ARGUMENTS];
	for (int i = 0; i < NUM_ARGUMENTS; ++i) {
		argsPtrs[i] = args[i].c_str();
	}

	ComPtr<IDxcIncludeHandler> includeHandler;
	if (FAILED(library->CreateIncludeHandler(&includeHandler))) {
		return {};
	}

	ComPtr<IDxcOperationResult> result;

	HRESULT hr = compiler->Compile(
		sourceBlob.Get(),
		name.c_str(),
		entryPoint.c_str(),
		target.c_str(),
		argsPtrs, NUM_ARGUMENTS,
		nullptr, 0, // defines
		includeHandler.Get(),
		result.GetAddressOf()
	);

	if (result) {
		result->GetStatus(&hr);
	}

	if (FAILED(hr)) {
		if (result) {
			ComPtr<IDxcBlobEncoding> errorsBlob;
			hr = result->GetErrorBuffer(&errorsBlob);
			if (SUCCEEDED(hr) && errorsBlob) {
				wprintf(L"Compilation failed with errors:\n\t%hs\n", (const char *)errorsBlob->GetBufferPointer());
			}
		}

		return {};
	}

	ShaderCompilerResult res = {};
	if (FAILED(result->GetResult(&res.binary))) {
		return {};
	}

	return res;
}

} // namespace Dar