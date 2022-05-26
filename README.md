# Dar

A wanna-be game engine for learning purposes.

## Description

This project was started as a playground for learning DirectX12 but soon started to grow into a simple framework for games. It's planned to grow into a small game engine for RTS games. Since it is only for learning purposes I don't restrict myself to supporting different platforms or hardware, thus I allow myself to use the latest APIs/available hardware.

## Getting Started

### Dependencies

* DirectX 12.1
* Windows 10 version 2004
* CMake 3.14
* GPU supporting DX12 Shader model 6.6 (or the so-called DirectX Ultimate).

### Installing

* Use CMake to generate a Visual Studio 2019+ project
  * Optionally select `DAR_COMPILE_SAMPLE_PROJECTS` to build the sample projects.
* The `ALL_BUILD` target would build all of the projects.
  * The `Dar` project builds the framework into a static library. You can test
  the current rendering capabilities of the engine with the `Sponza` sample.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Third-party

* [GLFW](https://github.com/glfw/glfw)
* [MikkTSpace](https://github.com/mmikk/MikkTSpace)
* [assimp](https://github.com/assimp/assimp)
* [ImGui](https://github.com/ocornut/imgui)
* [optick](https://github.com/bombomby/optick)
* [stb](https://github.com/nothings/stb)