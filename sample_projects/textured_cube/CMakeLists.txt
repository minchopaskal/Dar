set(TEX_CUBE_DIR ${PROJECT_SOURCE_DIR}/sample_projects/textured_cube)

set(INCLUDE_DIR ${TEX_CUBE_DIR}/include)
set(SRC_DIR ${TEX_CUBE_DIR}/src)
set(RESOURCES_DIR ${TEX_CUBE_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/fps_camera_controller.h
	${INCLUDE_DIR}/d3d12_tex_cube.h
	${INCLUDE_DIR}/geometry.h
)

set(SOURCES
	${SRC_DIR}/fps_camera_controller.cpp
	${SRC_DIR}/main.cpp
	${SRC_DIR}/d3d12_tex_cube.cpp
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

add_executable(TexturedCube ${HEADERS} ${SOURCES} ${SUBMODULES} ${SHADERS})

compile_shaders(
	TexturedCube
	"${SHADERS}"
	"6_6"
)

set_property(TARGET TexturedCube PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:TexturedCube>)

set_target_properties(TexturedCube PROPERTIES FOLDER SampleProjects)

target_compile_definitions(
	TexturedCube
	PRIVATE
	$<$<CONFIG:Debug>:DAR_DEBUG>
	$<$<CONFIG:Release>:DAR_NDEBUG>
)

target_compile_definitions(
	TexturedCube
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(TexturedCube PRIVATE ${LIB_DIR})

target_link_libraries(TexturedCube Dar "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	TexturedCube
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${PROJECT_SOURCE_DIR}/DarE
	${PROJECT_SOURCE_DIR}/DarE/graphics
	${SUBMODULE_DIR}/imgui
)

copyD3D12Core(TexturedCube)

add_custom_command(
	TARGET TexturedCube
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_DIR}/textures/ $<TARGET_FILE_DIR:TexturedCube>/res/textures/
)
