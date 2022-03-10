set(HELLO_TRI_DIR ${PROJECT_SOURCE_DIR}/sample_projects/hello_triangle)

set(INCLUDE_DIR ${HELLO_TRI_DIR}/include)
set(SRC_DIR ${HELLO_TRI_DIR}/src)
set(RESOURCES_DIR ${HELLO_TRI_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/d3d12_hello_triangle.h
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

add_executable(HelloTriangle ${HEADERS} ${SOURCES} ${SUBMODULES} ${SHADERS})

compile_shaders(HelloTriangle "${SHADERS}" "6_5")

set_property(TARGET HelloTriangle PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:HelloTriangle>)

set_target_properties(HelloTriangle PROPERTIES FOLDER SampleProjects)

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
	${PROJECT_SOURCE_DIR}/D3D12BaseLib/include
)

copyD3D12Core(HelloTriangle)
