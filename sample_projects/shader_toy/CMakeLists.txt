set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(RESOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/shader_toy.h
)

set(SOURCES
	${SRC_DIR}/main.cpp
	${SRC_DIR}/shader_toy.cpp
	${SUBMODULE_DIR}/imguifiledialog/ImGuiFileDialog.cpp
)

set(SHADERS
	${SHADERS_DIR}/common.hlsli
	${SHADERS_DIR}/render_pass_ps.hlsl
	${SHADERS_DIR}/render_pass_vs.hlsl
	${SHADERS_DIR}/screen_quad_ps.hlsl
	${SHADERS_DIR}/screen_quad_vs.hlsl
	${SHADERS_DIR}/screen_quad.hlsli
)

set(TEXT_EDITOR_SRC
	${SUBMODULE_DIR}/ImGuiColorTextEdit/TextEditor.cpp
)

source_group("src"           FILES ${SOURCES})
source_group("include"       FILES ${HEADERS})
source_group("shaders"       FILES ${SHADERS})
source_group("third_party"   FILES ${TEXT_EDITOR_SRC})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

add_executable(ShaderToy ${HEADERS} ${SOURCES} ${SUBMODULES} ${SHADERS} ${TEXT_EDITOR_SRC})

compile_shaders(ShaderToy "${SHADERS}" "6_6")

set_property(TARGET ShaderToy PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:ShaderToy>)

set_target_properties(ShaderToy PROPERTIES FOLDER SampleProjects)

target_compile_definitions(
	ShaderToy
	PRIVATE
	$<$<CONFIG:Debug>:DAR_DEBUG>
	$<$<CONFIG:Release>:DAR_NDEBUG>
)

target_compile_definitions(
	ShaderToy
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(ShaderToy PRIVATE ${LIB_DIR})

target_link_libraries(ShaderToy Dar "d3d12.lib" "dxcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	ShaderToy
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${PROJECT_SOURCE_DIR}/DarE
	${PROJECT_SOURCE_DIR}/DarE/graphics
	${SUBMODULE_DIR}/imgui
)

copyD3D12Core(ShaderToy)

set(SHADERS_TO_COPY
	${SHADERS_DIR}/common.hlsli
	${SHADERS_DIR}/screen_quad.hlsli
)

set(DXC_FILES_TO_COPY
	${SUBMODULE_DIR}/dxc_1.6/dxc.exe
	${SUBMODULE_DIR}/dxc_1.6/dxil.dll
	${SUBMODULE_DIR}/dxc_1.6/dxcompiler.dll
)

add_custom_command(TARGET ShaderToy POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:ShaderToy>/res/shaders
	COMMAND ${CMAKE_COMMAND} -E copy ${SHADERS_TO_COPY} $<TARGET_FILE_DIR:ShaderToy>/res/shaders
	COMMAND ${CMAKE_COMMAND} -E copy ${DXC_FILES_TO_COPY} $<TARGET_FILE_DIR:ShaderToy>
)
