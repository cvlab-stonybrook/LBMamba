cmake_minimum_required(VERSION 3.20)
project(LocallyBidirectionalParallelScan CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)


if (NOT $ENV{USER} STREQUAL "")
    if ($ENV{USER} STREQUAL "xihan1")
        if (NOT DEFINED Python_ROOT_DIR)
            set(Python_ROOT_DIR "$ENV{HOME}/opt/anaconda3/envs/2dmano")
        endif ()
    else ()
        # pass in -DPython_ROOT_DIR=/path/to/python
        # set(Python_ROOT_DIR "/opt/conda")
    endif ()
endif ()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
# libTorch conda build conflicts with OpenCV, so download compiled library directly from pytorch.org.
# libtorch REQUIRES CMAKE_CUDA_STANDARD <= 17 and CMAKE_CXX_STANDARD <= 17.
set(CAFFE2_USE_CUDNN ON)
set(Torch_ROOT_DIR "${Python_ROOT_DIR}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/torch")
message(${Torch_ROOT_DIR})
find_package(Torch REQUIRED CONFIG HINTS "${Torch_ROOT_DIR}/share/cmake")



set(TORCH_EXTENSION pscan)

add_library(${TORCH_EXTENSION} SHARED
        pscan.cu
        reverse_scan.cuh
        selective_scan.h
        selective_scan_bwd.cu
        selective_scan_fwd.cu
        selective_scan_bwd_fp16_real.cu
        selective_scan_bwd_fp32_real.cu
        selective_scan_bwd_kernel.cuh
        selective_scan_common.h
        selective_scan_fwd_fp16.cu
        selective_scan_fwd_fp32.cu
        selective_scan_fwd_kernel.cuh
        static_switch.h
        uninitialized_copy.cuh
)

#if (NOT $ENV{USER} STREQUAL "")
#    if ($ENV{USER} STREQUAL "xihan1")
#        set(${TORCH_EXTENSION}_OUTPUT_LIBRARY ${CMAKE_SOURCE_DIR}/mambapy/pscan_cuda/)
#    else ()
#        set(${TORCH_EXTENSION}_OUTPUT_LIBRARY ${CMAKE_SOURCE_DIR}/../v2dmamba_scan/)
#    endif()
#else ()
#    set(${TORCH_EXTENSION}_OUTPUT_LIBRARY ${CMAKE_SOURCE_DIR}/../v2dmamba_scan/)
#endif()

if (NOT $ENV{USER} STREQUAL "")
    if ($ENV{USER} STREQUAL "xihan1")
        if (NOT DEFINED OUTPUT_DIRECTORY)
            set(OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lbm/pscan_cuda/)
        endif ()
    else ()
        # pass in -DOUTPUT_DIRECTORY=/path/to/output
        # set(OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../v2dmamba_scan/)
    endif ()
else ()
    # pass in -DOUTPUT_DIRECTORY=/path/to/output
    # set(OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../v2dmamba_scan/)
endif()

if (NOT DEFINED CUDA_ARCHS)
    set(CUDA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
endif ()

set_target_properties(${TORCH_EXTENSION} PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${TORCH_EXTENSION}
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY}
        CUDA_ARCHITECTURES ${CUDA_ARCHS}
)

target_include_directories(${TORCH_EXTENSION} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${Torch_ROOT_DIR}/include
        ${Torch_ROOT_DIR}/include/torch/csrc/api/include
        ${Torch_ROOT_DIR}/include/TH
        ${Torch_ROOT_DIR}/include/THC
        ${Python_INCLUDE_DIRS}
#        /usr/local/cuda/include
)

#set(NAN_SMEM_CHECK ON)
#set(NAN_GRAD_CHECK ON)

if (DEFINED BOUNDARY_CHECK)
    target_compile_definitions(${TORCH_EXTENSION} PUBLIC -DBOUNDARY_CHECK)
endif ()

if (DEFINED NAN_SMEM_CHECK)
    target_compile_definitions(${TORCH_EXTENSION} PUBLIC -DNAN_SMEM_CHECK)
endif ()

if (DEFINED NAN_GRAD_CHECK)
    target_compile_definitions(${TORCH_EXTENSION} PUBLIC -DNAN_GRAD_CHECK)
endif ()

target_compile_definitions(${TORCH_EXTENSION} PUBLIC
        -D__CUDA_NO_HALF_OPERATORS__
        -D__CUDA_NO_HALF_CONVERSIONS__
        -D__CUDA_NO_HALF2_OPERATORS__
        -D__CUDA_NO_HALF2_CONVERSIONS__
        -D__CUDA_NO_BFLOAT16_OPERATORS__
        -D__CUDA_NO_BFLOAT16_CONVERSIONS__
        -D__CUDA_NO_BFLOAT162_OPERATORS__
        -D__CUDA_NO_BFLOAT162_CONVERSIONS__
)
target_compile_options(${TORCH_EXTENSION} PUBLIC
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xptxas -warn-spills#,-v
        # -O3 -NDEBUG is default in CMAKE_CXX_FLAGS_RELEASE and CMAKE_CUDA_FLAGS_RELEASE
        $<$<CONFIG:DEBUG>:-O0>
        # -g is default in CMAKE_CXX_FLAGS_DEBUG and CMAKE_CUDA_FLAGS_DEBUG
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>
        -lineinfo
        --compiler-options -fPIC
        -DTORCH_API_INCLUDE_EXTENSION_H
        -DPYBIND11_COMPILER_TYPE="_gcc"
        -DPYBIND11_STDLIB="_libstdcpp"
        -DPYBIND11_BUILD_ABI="_cxxabi1011"
        -D_GLIBCXX_USE_CXX11_ABI=0
        -DTORCH_EXTENSION_NAME=${TORCH_EXTENSION}
)

target_link_directories(${TORCH_EXTENSION} PUBLIC
        ${Python_LIBRARY_DIRS}
        ${Torch_ROOT_DIR}/lib
#        /usr/local/cuda/lib64
)
target_link_libraries(${TORCH_EXTENSION} PUBLIC
        c10
        c10_cuda
        cudart
        curand
        torch
        torch_cpu
        torch_cuda
        torch_python
)
target_link_options(${TORCH_EXTENSION} PUBLIC
        -pthread
        -B ${Python_ROOT_DIR}/compiler_compat
        -Wl,-rpath,${Python_LIBRARY_DIRS}
        -Wl,-rpath-link,${Python_LIBRARY_DIRS}
)


#set(TEST test)
#add_executable(${TEST}
#        include/scan/block_scan.cuh
#        include/scan/block_scan_warp_scans.cuh
#        include/scan/commons.h
#        include/scan/thread_reduce.cuh
#        include/scan/thread_scan.cuh
#        include/scan/warp_scan.cuh
#        include/scan/warp_scan_shfl.cuh
#        include/utils/cuda_utils.h
#        src/test_arr.cu
#)
#set_target_properties(${TEST} PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RESOLVE_DEVICE_SYMBOLS ON
#)
#target_compile_definitions(${TEST} PUBLIC
#        -D__CUDA_NO_HALF_OPERATORS__
#        -D__CUDA_NO_HALF_CONVERSIONS__
#        -D__CUDA_NO_HALF2_OPERATORS__
#        -D__CUDA_NO_HALF2_CONVERSIONS__
#        -D__CUDA_NO_BFLOAT16_OPERATORS__
#        -D__CUDA_NO_BFLOAT16_CONVERSIONS__
#        -D__CUDA_NO_BFLOAT162_OPERATORS__
#        -D__CUDA_NO_BFLOAT162_CONVERSIONS__
#        -D_GLIBCXX_USE_CXX11_ABI=0
#)
#target_compile_options(${TEST} PUBLIC
#        --use_fast_math
#        --expt-relaxed-constexpr
#        --expt-extended-lambda
#        -Xptxas -warn-spills#,-v
#        # -O3 -NDEBUG is default in CMAKE_CXX_FLAGS_RELEASE and CMAKE_CUDA_FLAGS_RELEASE
#        $<$<CONFIG:DEBUG>:-O0>
#        # -g is default in CMAKE_CXX_FLAGS_DEBUG and CMAKE_CUDA_FLAGS_DEBUG
#        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>
#        -lineinfo
#        --compiler-options -fPIC
#)
#target_include_directories(${TEST} PUBLIC
#        ${CMAKE_SOURCE_DIR}/include
#        /usr/local/cuda/include
#)
#target_link_libraries(${TEST}
#        cudart
#        curand
#)

# NOT NEEDED given CMAKE_CUDA_ARCHITECTURE is set properly
# target_compile_options(${EXECUTABLE} PRIVATE
#         $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=[compute_75,sm_75]>)