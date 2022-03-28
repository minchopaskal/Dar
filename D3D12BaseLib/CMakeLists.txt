set(LIB_SOURCE_DIR ${PROJECT_SOURCE_DIR}/D3D12BaseLib)

set(INCLUDE_DIR ${LIB_SOURCE_DIR}/include)
set(SRC_DIR ${LIB_SOURCE_DIR}/src)
set(RESOURCES_DIR ${LIB_SOURCE_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/d3d12_app.h
	${INCLUDE_DIR}/d3d12_asset_manager.h
	${INCLUDE_DIR}/d3d12_async.h
	${INCLUDE_DIR}/d3d12_camera.h
	${INCLUDE_DIR}/d3d12_command_list.h
	${INCLUDE_DIR}/d3d12_command_queue.h
	${INCLUDE_DIR}/d3d12_defines.h
	${INCLUDE_DIR}/d3d12_depth_buffer.h
	${INCLUDE_DIR}/d3d12_descriptor_heap.h
	${INCLUDE_DIR}/d3d12_includes.h
	${INCLUDE_DIR}/d3d12_input_query.h
	${INCLUDE_DIR}/d3d12_logger.h
	${INCLUDE_DIR}/d3d12_math.h
	${INCLUDE_DIR}/d3d12_pipeline_state.h
	${INCLUDE_DIR}/random.h
	${INCLUDE_DIR}/d3d12_resource_handle.h
	${INCLUDE_DIR}/d3d12_resource_manager.h
	${INCLUDE_DIR}/d3d12_scene.h
	${INCLUDE_DIR}/d3d12_timer.h
	${INCLUDE_DIR}/d3d12_utils.h
	${INCLUDE_DIR}/d3d12_vertex_index_buffer.h
)

set(SOURCES
	${SRC_DIR}/d3d12_app.cpp
	${SRC_DIR}/d3d12_asset_manager.cpp
	${SRC_DIR}/d3d12_camera.cpp
	${SRC_DIR}/d3d12_command_list.cpp
	${SRC_DIR}/d3d12_command_queue.cpp
	${SRC_DIR}/d3d12_depth_buffer.cpp
	${SRC_DIR}/d3d12_descriptor_heap.cpp
	${SRC_DIR}/d3d12_logger.cpp
	${SRC_DIR}/d3d12_pipeline_state.cpp
	${SRC_DIR}/d3d12_resource_manager.cpp
	${SRC_DIR}/d3d12_resource_handle.cpp
	${SRC_DIR}/d3d12_scene.cpp
	${SRC_DIR}/d3d12_utils.cpp

)

set(SUBMODULES
	${SUBMODULE_DIR}/d3dx12.h
)

set(IMGUI_DIR ${SUBMODULE_DIR}/imgui)

set(IMGUI_SRC
	${IMGUI_DIR}/imgui.cpp
	${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_tables.cpp
	${IMGUI_DIR}/imgui_widgets.cpp
	${IMGUI_DIR}/backends/imgui_impl_dx12.cpp
	${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
)

source_group("src"           FILES ${SOURCES})
source_group("include"       FILES ${HEADERS})
source_group("third_party"   FILES ${SUBMODULES} ${IMGUI_SRC})

add_library(D3D12BaseLib ${HEADERS} ${SOURCES} ${SUBMODULES} ${IMGUI_SRC})

set_property(TARGET D3D12BaseLib PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:D3D12BaseLib>)

target_compile_definitions(
	D3D12BaseLib
	PRIVATE
	$<$<CONFIG:Debug>:D3D12_DEBUG>
	$<$<CONFIG:Release>:D3D12_NDEBUG>
)

target_compile_definitions(
	D3D12BaseLib
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(D3D12BaseLib PRIVATE ${LIB_DIR})

target_link_libraries(D3D12BaseLib "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	D3D12BaseLib
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${SUBMODULE_DIR}/imgui
)
