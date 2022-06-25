cmake_minimum_required(VERSION 3.14)

project(Dar VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

# Enable parallel compilation for MSVC by defult.
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/we4834>) # Make an error ignoring result of nodiscard function
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/we4715>) # Make an error failing to provide each path with a return value.
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/wd4996>) # Ignore CRT non-conforming warnings.
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/wd4099>) # Ignore linking wihtout PBDs.
#add_compile_options($<$<CONFIG:Debug>:/WX>)

set(DAR_COMPILE_SAMPLE_PROJECTS CACHE BOOL "Compile sample projects")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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
		${DLLS_TO_COPY} $<TARGET_FILE_DIR:${TRG}>/D3D12/
	)
	#set(DLLS_TO_COPY
	#	${SUBMODULE_DIR}/dxc_1.6/dxil.dll
	#	${SUBMODULE_DIR}/dxc_1.6/dxcompiler.dll
	#)
	#add_custom_command(TARGET ${TRG} POST_BUILD
	#	COMMAND ${CMAKE_COMMAND} -E copy ${DLLS_TO_COPY} $<TARGET_FILE_DIR:${TRG}>/
	#)
endmacro()

macro(compile_shaders TRG SHADER_FILES SHADER_VERSION)
	add_custom_command(TARGET ${TRG} PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TRG}>/res/shaders/
	)

	if (TARGET CompileShaders)
	else()
		add_custom_target(CompileShaders ALL)
	endif()

	foreach(SHADER_FILE ${SHADER_FILES})
		get_filename_component(SHADERS_FOLDER ${SHADER_FILE} DIRECTORY)
		get_filename_component(SF_NAME ${SHADER_FILE} NAME_WE)
		get_filename_component(SF_NAME_EXT ${SHADER_FILE} NAME)

		string(FIND ${SF_NAME_EXT} ".hlsli" IS_HEADER_FILE)
		if (NOT ${IS_HEADER_FILE} STREQUAL "-1")
			message("Skipping ${SF_NAME_EXT}...")
			continue()
		endif()

		message("Setting properties of ${SF_NAME_EXT}")

		set(SHADER_TYPE "")
		string(FIND ${SHADER_FILE} "_ps.hlsl" IS_PIXEL_SHADER)
		string(FIND ${SHADER_FILE} "_vs.hlsl" IS_VERTEX_SHADER)
		string(FIND ${SHADER_FILE} "_gs.hlsl" IS_GEOM_SHADER)
		string(FIND ${SHADER_FILE} "_hs.hlsl" IS_HULL_SHADER)
		string(FIND ${SHADER_FILE} "_ms.hlsl" IS_MESH_SHADER)
		string(FIND ${SHADER_FILE} "_as.hlsl" IS_AMP_SHADER)
		string(FIND ${SHADER_FILE} "_ds.hlsl" IS_DOMAIN_SHADER)
		string(FIND ${SHADER_FILE} "_cs.hlsl" IS_COMPUTE_SHADER)
		if (NOT ${IS_PIXEL_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "ps_${SHADER_VERSION}")
		elseif(NOT ${IS_VERTEX_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "vs_${SHADER_VERSION}")
		elseif(NOT ${IS_GEOM_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "gs_${SHADER_VERSION}")
		elseif(NOT ${IS_HULL_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "hs_${SHADER_VERSION}")
		elseif(NOT ${IS_MESH_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "ms_${SHADER_VERSION}")
		elseif(NOT ${IS_AMP_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "as_${SHADER_VERSION}")
		elseif(NOT ${IS_DOMAIN_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "ds_${SHADER_VERSION}")
		elseif(NOT ${IS_COMPUTE_SHADER} STREQUAL "-1")
			set(SHADER_TYPE "ps_${SHADER_VERSION}")
		else()
			message(FATAL_ERROR "Invalid shader name - either it's not ps/vs or it's not a .hlsl file!")
		endif()

		# exclude hlsl files from build since VS would try to compile them with fxc
		set_source_files_properties(${SHADER_FILE} PROPERTIES VS_SETTINGS "ExcludedFromBuild=True")

		if (TARGET ${TRG}_CompileShaders)
		else()
			add_custom_target(${TRG}_CompileShaders DEPENDS ${SHADER_FILE})
			add_dependencies(CompileShaders ${TRG}_CompileShaders)
		endif()
		
		set_target_properties(${TRG}_CompileShaders PROPERTIES FOLDER ShaderCompilerCommands)
		add_custom_command(TARGET ${TRG}_CompileShaders PRE_BUILD
			COMMAND ${SUBMODULE_DIR}/dxc_1.6/dxc.exe
			-E main # shader entry point. Only main is valid. Each different type of shader must be in its own file.
			-T ${SHADER_TYPE} # target profile
			-Fo $<TARGET_FILE_DIR:${TRG}>/res/shaders/${SF_NAME}.bin # output file location
			/I ${SHADERS_FOLDER}
			$<$<CONFIG:Debug>:-Zi> # Output debug information, but only in DEBUG mode
			$<$<CONFIG:Debug>:-Fd> $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${TRG}>/res/shaders/${SF_NAME}.pdb> # output pdb file
			$<$<CONFIG:Debug>:-O0> # no optimizations in debug
			${SHADER_FILE} # the .hlsl file to be compiled
			COMMAND ${CMAKE_COMMAND} -E echo "Compiling ${SF_NAME_EXT}"
			DEPENDS ${SHADER_FILE}
			
		)

	endforeach()
endmacro()

add_subdirectory(DarE)
add_subdirectory(sample_projects)
