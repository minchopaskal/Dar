cmake_minimum_required(VERSION 3.14)

project(D3D12Dive VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(RESOURCES_DIR ${PROJECT_SOURCE_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(GLM_DIR ${SUBMODULE_DIR}/glm)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/defines.h
	${INCLUDE_DIR}/d3d12_app.h
	${INCLUDE_DIR}/d3d12_hello_triangle.h
	${INCLUDE_DIR}/win32_app.h
	${INCLUDE_DIR}/asset_manager.h
	${INCLUDE_DIR}/geometry.h
)

set(SOURCES
	${SRC_DIR}/main.cpp
	${SRC_DIR}/d3d12_hello_triangle.cpp
	${SRC_DIR}/win32_app.cpp
	${SRC_DIR}/asset_manager.cpp
)

# Include only the files one wants to be seen in the IDE
# Each file that must linked goes to SUBMODULES_INTERNAL
set(SUBMODULES
	${SUBMODULE_DIR}/d3dx12.h
)

set(SHADERS
	${SHADERS_DIR}/basic.hlsl
)

source_group("src"         FILES ${SOURCES})
source_group("include"     FILES ${HEADERS})
source_group("third_party" FILES ${SUBMODULES})
source_group("shaders"     FILES ${SHADERS})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

add_executable(D3D12Dive ${HEADERS} ${SOURCES} ${SUBMODULES}) # ${SHADERS})

set_property(TARGET D3D12Dive PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:D3D12Dive>)

target_compile_definitions(
	D3D12Dive
	PRIVATE
	$<$<CONFIG:Debug>:D3D12_DEBUG>
	$<$<CONFIG:RelWithDebInfo>:D3D12_DEBUG>
	$<$<CONFIG:Release>:D3D12_NDEBUG>
	$<$<CONFIG:MinSizeRel>:D3D12_NDEBUG>
)

target_compile_definitions(
	D3D12Dive
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(D3D12Dive PRIVATE ${LIB_DIR})

target_link_libraries(D3D12Dive "d3d12.lib" "dxgi.lib" "d3dcompiler.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	D3D12Dive
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${GLM_DIR}
)

add_custom_command(TARGET D3D12Dive PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	${RESOURCES_DIR} $<TARGET_FILE_DIR:D3D12Dive>/res/
)
