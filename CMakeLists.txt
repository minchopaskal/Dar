cmake_minimum_required(VERSION 3.14)

project(D3D12Base VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

# Enable parallel compilation for MSVC by defult.
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

set(D3D12_COMPILE_SAMPLE_PROJECTS CACHE BOOL "Compile sample projects")

macro(copyD3D12Core TRG)
	set(DLLS_TO_COPY
		${LIB_DIR}/D3D12AgilitySDK/D3D12Core.dll
		$<$<CONFIG:Debug>:${LIB_DIR}/D3D12AgilitySDK/D3D12Core.pdb>
		$<$<CONFIG:Debug>:${LIB_DIR}/D3D12AgilitySDK/D3D12SDKLayers.dll>
		$<$<CONFIG:Debug>:${LIB_DIR}/D3D12AgilitySDK/D3D12SDKLayers.pdb>
	)
	add_custom_command(TARGET ${TRG} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TRG}>/D3D12/
		COMMAND ${CMAKE_COMMAND} -E copy
		${DLLS_TO_COPY} $<TARGET_FILE_DIR:${TRG}>/D3D12/)
endmacro()

# Temporary solution until shader model 6.6 is supported in windows.
macro(compile_shaders TRG SHADER_FOLDER)
	add_custom_command(TARGET ${TRG} PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TRG}>/res/shaders/
	)
	foreach(SHADER_FILE ${SHADERS})
		get_filename_component(SF_NAME ${SHADER_FILE} NAME_WE)
		get_filename_component(SF_NAME_EXT ${SHADER_FILE} NAME)
		message("Setting properties of ${SF_NAME_EXT}")

		set(SHADER_VERSION "")
		string(FIND ${SHADER_FILE} "_ps.hlsl" IS_PIXEL_SHADER)
		string(FIND ${SHADER_FILE} "_vs.hlsl" IS_VERTEX_SHADER)
		string(FIND ${SHADER_FILE} "_gs.hlsl" IS_GEOM_SHADER)
		string(FIND ${SHADER_FILE} "_hs.hlsl" IS_HULL_SHADER)
		string(FIND ${SHADER_FILE} "_ms.hlsl" IS_MESH_SHADER)
		string(FIND ${SHADER_FILE} "_as.hlsl" IS_AMP_SHADER)
		string(FIND ${SHADER_FILE} "_ds.hlsl" IS_DOMAIN_SHADER)
		string(FIND ${SHADER_FILE} "_cs.hlsl" IS_COMPUTE_SHADER)
		if (NOT ${IS_PIXEL_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "ps_6_6")
		elseif(NOT ${IS_VERTEX_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "vs_6_6")
		elseif(NOT ${IS_GEOM_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "gs_6_6")
		elseif(NOT ${IS_HULL_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "hs_6_6")
		elseif(NOT ${IS_MESH_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "ms_6_6")
		elseif(NOT ${IS_AMP_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "as_6_6")
		elseif(NOT ${IS_DOMAIN_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "ds_6_6")
		elseif(NOT ${IS_COMPUTE_SHADER} STREQUAL "-1")
			set(SHADER_VERSION "ps_6_6")
		else()
			message(FATAL_ERROR "Invalid shader name - either it's not ps/vs or it's not a .hlsl file!")
		endif()

		# exclude hlsl files from build since VS would try to compile them with fxc
		set_source_files_properties(${SHADER_FILE} PROPERTIES VS_SETTINGS "ExcludedFromBuild=True")

		add_custom_command(TARGET ${TRG} PRE_BUILD
			COMMAND ${SUBMODULE_DIR}/dxc_1.6/dxc.exe
			-E main # shader entry point. Only main is valid. Each different type of shader must be in its own file.
			-T ${SHADER_VERSION} # target profile
			-Fo $<TARGET_FILE_DIR:${TRG}>/res/shaders/${SF_NAME}.bin # output file location
			$<$<CONFIG:Debug>:-Zi> # Output debug information, but only in DEBUG mode
			$<$<CONFIG:Debug>:-Fd> $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${TRG}>/res/shaders/${SF_NAME}.pdb> # output pdb file
			$<$<CONFIG:Debug>:-O0> # no optimizations in debug
			${SHADER_FILE} # the .hlsl file to be compiled
			COMMAND ${CMAKE_COMMAND} -E echo "Compiling ${SF_NAME_EXT}"
			DEPENDS ${SHADER_FILE}
		)
	endforeach()
endmacro()

add_subdirectory(D3D12BaseLib)
add_subdirectory(sample_projects)
