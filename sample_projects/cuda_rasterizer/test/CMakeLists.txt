set(CUDA_RAST_TEST_DIR ${PROJECT_SOURCE_DIR}/sample_projects/cuda_rasterizer/test)

set(INCLUDE_DIR ${CUDA_RAST_TEST_DIR}/include)
set(SRC_DIR ${CUDA_RAST_TEST_DIR}/src)
set(RESOURCES_DIR ${CUDA_RAST_TEST_DIR}/res)
set(GPU_DIR ${RESOURCES_DIR}/gpu)
set(SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/third_party)
set(PROJECT_SPECIFIC_SUBMODULE_DIR ${PROJECT_SOURCE_DIR}/sample_projects/cuda_rasterizer/third_party)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)
set(CUDA_RAST_BASE_DIR ${PROJECT_SOURCE_DIR}/sample_projects/cuda_rasterizer)
set(CUDA_RAST_BASE_INCLUDE ${CUDA_RAST_BASE_DIR}/include)

set(HEADERS
	${INCLUDE_DIR}/cuda_drawable.h
)

set(SRCS
	${SRC_DIR}/cuda_drawable.cpp
	${SRC_DIR}/main.cpp
)

set(CUDA_FILES
	${GPU_DIR}/basic_shader.cu
)

source_group("src"     FILES ${SRCS})
source_group("include" FILES ${HEADERS})
source_group("gpu"     FILES ${CUDA_FILES})

# Create stub library, so the .cu files show in the VS filters
add_custom_target(CudaRasterizerTestGPU ALL SOURCES ${CUDA_FILES})
set_target_properties(CudaRasterizerTestGPU PROPERTIES FOLDER SampleProjects/CudaRasterizer/Test)
set_source_files_properties(${GPU_DIR}/basic_shader.cuh PROPERTIES HEADER_FILE_ONLY ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")

add_executable(CudaRasterizerTest ${HEADERS} ${SRCS})
add_dependencies(CudaRasterizerTest CudaRasterizer)

set_property(TARGET CudaRasterizerTest PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:CudaRasterizerTest>)
set_target_properties(CudaRasterizerTest PROPERTIES FOLDER SampleProjects/CudaRasterizer/Test)

target_compile_definitions(
	CudaRasterizerTest
	PRIVATE
	$<$<CONFIG:Debug>:D3D12_DEBUG>
	$<$<CONFIG:Release>:D3D12_NDEBUG>
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(CudaRasterizerTest PRIVATE ${LIB_DIR})
target_link_directories(CudaRasterizerTest PRIVATE ${PROJECT_SPECIFIC_SUBMODULE_DIR}/lib)

target_link_libraries(CudaRasterizerTest CudaRasterizer D3D12BaseLib CUDABaseLib "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" "glfw3.lib")

target_include_directories(
	CudaRasterizerTest
	PRIVATE
	${INCLUDE_DIR}
	${SUBMODULE_DIR}
	${PROJECT_SOURCE_DIR}/D3D12BaseLib/include
	${SUBMODULE_DIR}/imgui
	${PROJECT_SPECIFIC_SUBMODULE_DIR}/CUDAPlayground/CUDABase/include
	${CUDA_RAST_BASE_INCLUDE}
)

cudaRasterizerPrepare(CudaRasterizerTest)

# copy object files to bin dir
add_custom_command(
	TARGET CudaRasterizerTest POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:CudaRasterizerTest>/res/obj
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_DIR}/obj $<TARGET_FILE_DIR:CudaRasterizerTest>/res/obj
)

find_package(CUDAToolkit)
# Include dir added to NVCC include path in order for it to see cuda_cpu_common.h and rasterizer_utils.cuh
set(NVCC_ADDITIONAL_ARGS 
	-I${CUDA_RAST_BASE_INCLUDE}
	-I${CUDA_RAST_BASE_DIR}/res/gpu
)

compilePtx(
	CudaRasterizerTest
	"${CUDA_FILES}"
	"${NVCC_ADDITIONAL_ARGS}"
	TRUE
)
