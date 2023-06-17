# Shader Toy

## Goal

ShaderToy wanna-be with a DirectX12 back-end.

## Features

* Write HLSL pixel shaders with shader model 6.6.
* Save/Load shaders to/from file.
* Recompile shaders on the fly.
* Add as much render passes to the pipeline as you want.
* Compile the passes in a render graph by specifying each pass's dependencies.
* ~~Add multiple render targets to each pass.~~ (WIP)
* Access to a `common.hlsli` header which contains:
	* `ConstantData` structure which gives some commonly used variables:
    	* Render width and height;
    	* Rendered frames count;
    	* Delta time;
    	* Random seed (see below).
	* `PSInput` structure containing the input to all the shaders you would write. It only contains a `float2 uv` member.
	* Function for reading the previous frame's render texture
	* Functions for generation random numbers given a seed. Possible use include passing the random seed from `ConstantData` multiplied by the uv coordinates.
* Ability to pause/restart the rendering if it uses the `ConstantData::frame` member for updating its state.
* Ability to add texture resources.
* Bonus: implementation of the Belousov–Zhabotinsky reaction simulation. Can be loaded from `res/shaders/belousov_zhabotinsky.hlsl`

## Usage

Just write the HLSLs code as you would a normal pixel shader. See `res/shaders/common.hlsli` for some common function as well as how to access the previous frame's render buffer.

Define a render graph by writing multiple shaders. If you want `Shader1` to use the render target of `Shader0` as a resource add `Shader0` as a dependency to `Shader1`. Such resources can be accessed via `ResourceDescriptorHeap[idx]` where `idx` is the 1-based index of the dependency you want to access. The 0-th index is reserved for the previous frame render buffer as you might want to reuse it.

If you have added texture resources their indices start from `N` where `N` is the number of dependencies the current shader has.

## Screenshot

![screenshot](res/github/shader%20toy.png)