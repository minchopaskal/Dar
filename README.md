# Dar

A wanna-be game engine for learning purposes.

## Description

This project was started as a playground for learning DirectX12 but soon started to grow into a simple framework for games. It's planned to grow into a small game engine for RTS games. Since it is only for learning purposes I don't restrict myself to supporting different platforms or hardware, thus I allow myself to use the latest APIs/available hardware.

## Getting Started

### Dependencies

* DirectX 12.1+
* Windows 10 version 2004+
* Visual Studio 2022
* GPU supporting DX12 Shader model 6.6+

### Installing

* Use the generated solution for Visual Studio 2022 located at `solution\`
	* OR use [Sharpmake](https://github.com/ubisoft/Sharpmake) to generate a solution for your Visual Studio version.
* The `Dar` project builds the framework into a static library. You can test the current rendering capabilities of the engine with the `Sponza` sample.
* Resources(such as shaders and textures) are compiled via the resourcecompiler.exe located at `tools\resourcecompiler\`. The resource compiler can be build via the `soltion\tools\toolssolution.sln`. Examples come with `.bat` scripts for building the resources.
	* Note that projects expect a certain file structure for the resources. They should be placed inside `res\` folder, shaders should be inside `res\shaders`, same for textures. See ResourceManagerLib::serde for more details

### Examples

* [HelloTriangle](examples/hello_triangle)
	* Render a simple triangle
* [ShaderPlaything](examples/shader_plaything)
	* An app in the likes of [ShaderToy](https://www.shadertoy.com/). Write HLSL shaders or load them from file. Supports simple render graph. See [common.hlsli](examples/shader_plaything/res/shaders/common.hlsli) for "built-ins" like in ShaderToy.
* [Sponza](examples/sponza)
	* A simple app loading the [Sponza scene](https://www.cryengine.com/marketplace/product/crytek/sponza-sample-scene) for testing different rendering features. Currently written:
		- Mip-mapping + block compression so textures are read as binary data and loaded directly to GPU memory
		- Defferred rendering
		- PBR materials
		- Shadow mapping (for directional and spot lights. 1 cascade + PCF)
		- Post-processing - fxaa, tone-mapping, gamma-correction
		- Simple HUD rendering

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Third-party

* [GLFW](https://github.com/glfw/glfw)
* [MikkTSpace](https://github.com/mmikk/MikkTSpace)
* [assimp](https://github.com/assimp/assimp)
* [ImGui](https://github.com/ocornut/imgui)
* [optick](https://github.com/bombomby/optick)
* [stb](https://github.com/nothings/stb)
* [nlohmann-json](https://github.com/nlohmann/json)