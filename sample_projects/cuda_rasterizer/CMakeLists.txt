set(CUDA_RAST_DIR ${PROJECT_SOURCE_DIR}/sample_projects/cuda_rasterizer)

set(INCLUDE_DIR ${CUDA_RAST_DIR}/include)
set(SRC_DIR ${CUDA_RAST_DIR}/src)
set(RESOURCES_DIR ${CUDA_RAST_DIR}/res)
set(SHADERS_DIR ${RESOURCES_DIR}/shaders)
set(GPU_DIR ${RESOURCES_DIR}/gpu)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(PROJECT_SPECIFIC_SUBMODULE_DIR ${CUDA_RAST_DIR}/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
	${INCLUDE_DIR}/d3d12_cuda_rasterizer.h
	${INCLUDE_DIR}/cuda_cpu_common.h
)

set(SRCS
	${SRC_DIR}/d3d12_cuda_rasterizer.cpp
)

set(SHADERS
	${SHADERS_DIR}/screen_quad_vs.hlsl
	${SHADERS_DIR}/screen_quad_ps.hlsl
)

set(CUDA_FILES
	${GPU_DIR}/rasterizer.cu
	${GPU_DIR}/rasterizer_utils.cu
	${GPU_DIR}/rasterizer_utils.cuh
)

source_group("src"     FILES ${SRCS})
source_group("include" FILES ${HEADERS})
source_group("shaders" FILES ${SHADERS})
source_group("gpu"     FILES ${CUDA_FILES})

# Create stub library, so the .cu files show in the VS filters
add_custom_target(CudaRasterizerGPU ALL SOURCES ${CUDA_FILES})
set_target_properties(CudaRasterizerGPU PROPERTIES FOLDER SampleProjects/CudaRasterizer)
set_source_files_properties(${GPU_DIR}/rasterizer.cu PROPERTIES HEADER_FILE_ONLY ON)
set_source_files_properties(${GPU_DIR}/rasterizer_utils.cu PROPERTIES HEADER_FILE_ONLY ON)
set_source_files_properties(${GPU_DIR}/rasterizer_utils.cuh PROPERTIES HEADER_FILE_ONLY ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

add_library(CudaRasterizer ${HEADERS} ${SRCS} ${SHADERS})
add_dependencies(CudaRasterizer CUDABaseLib)

compile_shaders(CudaRasterizer "${SHADERS}" "6_6")

set_property(TARGET CudaRasterizer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:CudaRasterizer>)
set_target_properties(CudaRasterizer PROPERTIES FOLDER SampleProjects/CudaRasterizer)

target_compile_definitions(
	CudaRasterizer
	PRIVATE
	$<$<CONFIG:Debug>:DAR_DEBUG>
	$<$<CONFIG:Release>:DAR_NDEBUG>
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(CudaRasterizer PRIVATE ${LIB_DIR})
target_link_directories(CudaRasterizer PRIVATE ${PROJECT_SPECIFIC_SUBMODULE_DIR}/lib)

target_link_libraries(CudaRasterizer Dar CUDABaseLib "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

add_subdirectory(third_party)

target_include_directories(
	CudaRasterizer
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${PROJECT_SOURCE_DIR}/DarE
	${PROJECT_SOURCE_DIR}/DarE/graphics
	${SUBMODULE_DIR}/imgui
	${PROJECT_SPECIFIC_SUBMODULE_DIR}/CUDAPlayground/CUDABase/include
)

find_package(CUDAToolkit)
# Include dir added to NVCC include path in order for it to see cuda_cpu_common.h
set(NVCC_ADDITIONAL_ARGS 
	-I${INCLUDE_DIR}
)
compilePtx(
	CudaRasterizer
	"${CUDA_FILES}"
	"${NVCC_ADDITIONAL_ARGS}"
	TRUE
)

macro(cudaRasterizerPrepare TRG)
copyD3D12Core(${TRG})

add_custom_command(
	TARGET ${TRG} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TRG}>/data
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TRG}>/res/shaders
	COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:CudaRasterizer>/data $<TARGET_FILE_DIR:${TRG}>/data
	COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:CudaRasterizer>/res/shaders $<TARGET_FILE_DIR:${TRG}>/res/shaders
)
endmacro()

add_subdirectory(test)
