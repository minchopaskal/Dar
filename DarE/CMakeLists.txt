set(LIB_SOURCE_DIR ${PROJECT_SOURCE_DIR}/DarE)

set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${LIB_SOURCE_DIR}/asset_manager/asset_manager.h
	${LIB_SOURCE_DIR}/async/async.h
	${LIB_SOURCE_DIR}/async/job_system.h
	${LIB_SOURCE_DIR}/framework/app.h
	${LIB_SOURCE_DIR}/framework/camera.h
	${LIB_SOURCE_DIR}/framework/input_query.h
	${LIB_SOURCE_DIR}/graphics/d3d12/command_list.h
	${LIB_SOURCE_DIR}/graphics/d3d12/command_queue.h
	${LIB_SOURCE_DIR}/graphics/d3d12/data_buffer.h
	${LIB_SOURCE_DIR}/graphics/d3d12/depth_buffer.h
	${LIB_SOURCE_DIR}/graphics/d3d12/descriptor_heap.h
	${LIB_SOURCE_DIR}/graphics/d3d12/includes.h
	${LIB_SOURCE_DIR}/graphics/d3d12/pipeline_state.h
	${LIB_SOURCE_DIR}/graphics/d3d12/resource_handle.h
	${LIB_SOURCE_DIR}/graphics/d3d12/resource_manager.h
	${LIB_SOURCE_DIR}/graphics/d3d12/texture_res.h
	${LIB_SOURCE_DIR}/graphics/d3d12/vertex_index_buffer.h
	${LIB_SOURCE_DIR}/graphics/renderer.h
	${LIB_SOURCE_DIR}/math/dar_math.h
	${LIB_SOURCE_DIR}/utils/defines.h
	${LIB_SOURCE_DIR}/utils/logger.h
	${LIB_SOURCE_DIR}/utils/profile.h
	${LIB_SOURCE_DIR}/utils/random.h
	${LIB_SOURCE_DIR}/utils/shader_compiler.h
	${LIB_SOURCE_DIR}/utils/timer.h
	${LIB_SOURCE_DIR}/utils/utils.h
)

set(SOURCES
	${LIB_SOURCE_DIR}/asset_manager/asset_manager.cpp
	${LIB_SOURCE_DIR}/async/job_system.cpp
	${LIB_SOURCE_DIR}/framework/app.cpp
	${LIB_SOURCE_DIR}/framework/camera.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/command_list.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/command_queue.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/data_buffer.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/depth_buffer.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/descriptor_heap.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/pipeline_state.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/resource_manager.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/resource_handle.cpp
	${LIB_SOURCE_DIR}/graphics/d3d12/texture_res.cpp
	${LIB_SOURCE_DIR}/graphics/renderer.cpp
	${LIB_SOURCE_DIR}/utils/logger.cpp
	${LIB_SOURCE_DIR}/utils/shader_compiler.cpp
	${LIB_SOURCE_DIR}/utils/utils.cpp
)

set(SUBMODULES
	${SUBMODULE_DIR}/d3dx12.h
)

set(IMGUI_DIR ${SUBMODULE_DIR}/imgui)

set(OPTICK_DIR ${SUBMODULE_DIR}/optick/src)

set(IMGUI_SRC
	${IMGUI_DIR}/imgui.cpp
	${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_tables.cpp
	${IMGUI_DIR}/imgui_widgets.cpp
	${IMGUI_DIR}/backends/imgui_impl_dx12.cpp
	${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
)

set(OPTICK_SRC
	${OPTICK_DIR}/optick_capi.cpp
	${OPTICK_DIR}/optick_core.cpp
	${OPTICK_DIR}/optick_gpu.cpp
	${OPTICK_DIR}/optick_gpu.d3d12.cpp
	${OPTICK_DIR}/optick_message.cpp
	${OPTICK_DIR}/optick_miniz.cpp
	${OPTICK_DIR}/optick_serialization.cpp
	${OPTICK_DIR}/optick_server.cpp
)

source_group("src"         FILES ${SOURCES})
source_group("include"     FILES ${HEADERS})
source_group("third_party" FILES ${SUBMODULES} ${IMGUI_SRC} ${OPTICK_SRC})

add_library(Dar ${HEADERS} ${SOURCES} ${SUBMODULES} ${IMGUI_SRC} ${OPTICK_SRC})

set_property(TARGET Dar PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:Dar>)

target_compile_definitions(
	Dar
	PRIVATE
	$<$<CONFIG:Debug>:DAR_DEBUG>
	$<$<CONFIG:Debug>:DAR_PROFILE>
	$<$<CONFIG:Release>:DAR_NDEBUG>
)

target_compile_definitions(
	Dar
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(Dar PRIVATE ${LIB_DIR})

target_link_libraries(Dar "d3d12.lib" "dxcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	Dar
	PRIVATE
	${LIB_SOURCE_DIR}
	${LIB_SOURCE_DIR}/graphics
	${SUBMODULE_DIR}
	${SUBMODULE_DIR}/imgui
	${SUBMODULE_DIR}/dxc_1.6
	${OPTICK_DIR}
)
