list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})

add_executable(GridTest grid.cu)
target_include_directories(GridTest PUBLIC
	"../include"
	"../dependencies"
	"../dependencies/catch-2/include"
	"../dependencies/cutlass/include"
	"../dependencies/cutlass/tools/util/include"
	"../dependencies/fmt/include"
	)
target_compile_definitions(GridTest PUBLIC ${TCNN_DEFINITIONS})
target_compile_options(GridTest PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(GridTest tiny-cuda-nn)
add_test(NAME GridTest COMMAND GridTest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
