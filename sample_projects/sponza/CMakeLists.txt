set(SPONZA_DIR ${PROJECT_SOURCE_DIR}/sample_projects/sponza)

set(INCLUDE_DIR ${SPONZA_DIR}/include)
set(SRC_DIR ${SPONZA_DIR}/src)
set(RESOURCES_DIR ${SPONZA_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/fps_camera_controller.h
	${INCLUDE_DIR}/sponza.h
	${INCLUDE_DIR}/scene_loader.h
)

set(SOURCES
	${SRC_DIR}/fps_camera_controller.cpp
	${SRC_DIR}/main.cpp
	${SRC_DIR}/scene_loader.cpp
	${SRC_DIR}/sponza.cpp
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

add_executable(Sponza ${HEADERS} ${SOURCES} ${SUBMODULES} ${SHADERS})

compile_shaders(
	Sponza
	${SHADERS_DIR}
	"6_6"
)

set_property(TARGET Sponza PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:Sponza>)

set_target_properties(Sponza PROPERTIES FOLDER SampleProjects)

target_compile_definitions(
	Sponza
	PRIVATE
	$<$<CONFIG:Debug>:D3D12_DEBUG>
	$<$<CONFIG:Release>:D3D12_NDEBUG>
)

target_compile_definitions(
	Sponza
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(Sponza PRIVATE ${LIB_DIR} )

target_link_libraries(Sponza D3D12BaseLib "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_link_libraries(
	D3D12BaseLib
	$<$<CONFIG:Debug>:assimpd>
	$<$<CONFIG:Release>:assimp>
)

target_include_directories(
	Sponza
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${PROJECT_SOURCE_DIR}/D3D12BaseLib/include
	${SUBMODULE_DIR}/imgui
	${SUBMODULE_DIR}/assimp/include
)

copyD3D12Core(Sponza)

add_custom_command(
	TARGET Sponza
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_DIR}/textures/ $<TARGET_FILE_DIR:Sponza>/res/textures/
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_DIR}/scenes/ $<TARGET_FILE_DIR:Sponza>/res/scenes/
)
