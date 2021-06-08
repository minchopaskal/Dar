set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SAMPLE_PROJECT_DIR ${PROJECT_SOURCE_DIR}/sample_projects/hello_triangle)

set(INCLUDE_DIR ${SAMPLE_PROJECT_DIR}/include)
set(SRC_DIR ${SAMPLE_PROJECT_DIR}/src)
set(RESOURCES_DIR ${SAMPLE_PROJECT_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${SAMPLE_PROJECT_DIR}/third_party)
set(GLM_DIR ${SUBMODULE_DIR}/glm)
set(LIB_DIR ${SAMPLE_PROJECT_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/d3d12_hello_triangle.h
	${INCLUDE_DIR}/defines.h
	${INCLUDE_DIR}/geometry.h
)

set(SOURCES
	${SRC_DIR}/main.cpp
	${SRC_DIR}/d3d12_hello_triangle.cpp
)

set(SHADERS
	${SHADERS_DIR}/basic_ps.hlsl
	${SHADERS_DIR}/basic_vs.hlsl
)

source_group("src"           FILES ${SOURCES})
source_group("include"       FILES ${HEADERS})
source_group("shaders"       FILES ${SHADERS})
source_group("third_party"   FILES ${SUBMODULES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

foreach(SHADER_FILE ${SHADERS})
	message("Setting properties of ${SHADER_FILE}")

	string(FIND ${SHADER_FILE} "_ps.hlsl" IS_PIXEL_SHADER)
	string(FIND ${SHADER_FILE} "_vs.hlsl" IS_VERTEX_SHADER)
	if (NOT ${IS_PIXEL_SHADER} STREQUAL "-1")
		set_source_files_properties(${SHADER_FILE} PROPERTIES VS_SHADER_TYPE "Pixel")
	elseif(NOT ${IS_VERTEX_SHADER} STREQUAL "-1")
		set_source_files_properties(${SHADER_FILE} PROPERTIES VS_SHADER_TYPE "Vertex")
	else()
		message(FATAL_ERROR "Invalid shader name - either it's not ps/vs or it's not a .hlsl file!")
	endif()

	set_source_files_properties(${SHADER_FILE} PROPERTIES VS_SHADER_MODEL "5.1")

	# Using the VS's generating expresions here to avoid nasty string manipulation in CMake
	set_source_files_properties(${SHADER_FILE} PROPERTIES VS_SHADER_OBJECT_FILE_NAME "$(OutDir)/res/shaders/%(Filename).cso")
endforeach()

add_executable(HelloTriangle ${HEADERS} ${SOURCES} ${SUBMODULES} ${SHADERS})

set_property(TARGET HelloTriangle PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:HelloTriangle>)

target_compile_definitions(
	HelloTriangle
	PRIVATE
	$<$<CONFIG:Debug>:D3D12_DEBUG>
	$<$<CONFIG:Release>:D3D12_NDEBUG>
)

target_compile_definitions(
	HelloTriangle
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(HelloTriangle PRIVATE ${LIB_DIR})

target_link_libraries(HelloTriangle D3D12BaseLib "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	HelloTriangle
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${GLM_DIR}
	${PROJECT_SOURCE_DIR}/D3D12BaseLib/include
)

add_custom_command(TARGET HelloTriangle PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	${RESOURCES_DIR} $<TARGET_FILE_DIR:HelloTriangle>/res/
)
