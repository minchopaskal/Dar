# Shader Toy

## Goal

ShaderToy wanna-be with a DirectX12 back-end.

## Features

* Write HLSL pixel shaders with shader model 6.6.
* Save/Load shaders to/from file.
* Recompile shaders on the fly.
* Add as much render passes to the pipeline as you want.
* Compile the passes in a render graph by specifying each pass's dependancies.
* Add multiple render targets to each pass.
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
* Bonus: implementation of the Belousov–Zhabotinsky reaction simulation. Can be loaded from `res/shaders/belousov_zhabotinsky.hlsl`

## Not done / not planned.

* Posibility of adding UAV resources.

## Screenshot

![screenshot](res/github/shader%20toy.png)