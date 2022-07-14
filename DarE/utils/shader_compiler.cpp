#include "shader_compiler.h"

#include <dxcapi.h>
#include <filesystem>
#include <fstream>

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

bool ShaderCompiler::compileFromFile(const WString &filename, const WString &outputDir, ShaderType type) {
	auto p = std::filesystem::absolute(filename.c_str());
	if (!std::filesystem::exists(p) || !std::filesystem::is_regular_file(p)) {
		return false;
	}

	auto name = p.filename();
	std::ifstream ifs(p.c_str(), std::ios::in|std::ios::ate);
	if (!ifs.good()) {
		return false;
	}

	auto size = ifs.tellg();
	char *memblock = new char[size];
	ifs.seekg(0, std::ios::beg);
	ifs.read(memblock, size);
	ifs.close();
	
	bool res = compileFromSource(memblock, name, outputDir, type);
	
	delete[] memblock;

	return res;
}

bool ShaderCompiler::compileFromSource(const char *src, const WString &name, const WString &outputDir, ShaderType type) {
	WString entryPoint = L"main";
	WString target = shaderTypeToStr(type) + L"_6_6";

	WString filename = name + L"TEMP";
	HANDLE f = CreateFileW(filename.c_str(), GENERIC_WRITE, FILE_SHARE_READ, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
	if (!f) {
		return false;
	}

	DWORD written = 0;
	DWORD toWrite = strlen(src);
	while (toWrite > 0) {
		WriteFile(f, src, toWrite, &written, 0);
		toWrite -= written;
		src += written;
	}

	CloseHandle(f);

	constexpr int NUM_ARGUMENTS = 9;
	WString args[NUM_ARGUMENTS];

	auto relativePath = std::filesystem::relative(outputDir + L"\\" + name + L"_" + shaderTypeToStr(type) + L".bin");
	auto outputPath = std::filesystem::absolute(relativePath);
	args[0] = L"dxc.exe";
	args[1] = L"-I " + WString(relativePath.parent_path().c_str());
	args[2] = L"-Fo";
	args[3] = outputPath.c_str();
	args[4] = L"-E";
	args[5] = L"main";
	args[6] = L"-T";
	args[7] = target.c_str();
	args[8] = filename;

	WString cmdLine;
	for (int i = 0; i < NUM_ARGUMENTS; ++i) {
		cmdLine += args[i];
		if (i < NUM_ARGUMENTS - 1) {
			cmdLine += L" ";
		}
	}

	// additional information
	STARTUPINFOW si;
	PROCESS_INFORMATION pi;

	// set the size of the structures
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// start the program up
	BOOL res = CreateProcessW(L"dxc.exe",   // the path
		cmdLine.data(),        // Command line
		NULL,           // Process handle not inheritable
		NULL,           // Thread handle not inheritable
		FALSE,          // Set handle inheritance to FALSE
		0,              // No creation flags
		NULL,           // Use parent's environment block
		NULL,           // Use parent's starting directory 
		&si,            // Pointer to STARTUPINFO structure
		&pi             // Pointer to PROCESS_INFORMATION structure (removed extra parentheses)
	);

	if (!res) {
		DWORD lastErr = GetLastError();
		LOG_FMT(Error, "DXC failed to start. Error: %d", lastErr);
		return false;
	}

	WaitForSingleObject(pi.hProcess, INFINITE);

	DWORD exitCode;
	GetExitCodeProcess(pi.hProcess, &exitCode);
	if (exitCode != 0) {
		LOG_FMT(Error, "DXC failed to compile %s. Error: %lu", name.c_str(), exitCode);
		return false;
	}

	// Close process and thread handles. 
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);

	DeleteFileW(filename.c_str());

	return true;
}

} // namespace Dar