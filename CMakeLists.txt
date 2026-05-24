# =============================================================================
# CMakeLists.txt — CUDA Ray Tracer
#
# Requirements:
#   • CMake >= 3.18
#   • CUDA Toolkit (nvcc must be on PATH)
#
# Common CUDA architectures:
#   GTX 10xx  → sm_61      RTX 20xx  → sm_75
#   RTX 30xx  → sm_86      RTX 40xx  → sm_89
#   GTX 16xx  → sm_75      A100      → sm_80
#
# Quick start:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#   cd build && bin/cpu_raytracer
#   cd build && bin/cuda_raytracer
#
# Override architecture:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(cuda_raytracer LANGUAGES CXX CUDA)

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA architecture — auto-detect preferred, fallback to sm_75 ──────────────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # "native" tells CMake/nvcc to query the installed GPU automatically
    # (requires CMake 3.24+ for the "native" keyword; falls back gracefully)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
        message(STATUS "CMake < 3.24 — defaulting to sm_75. "
                       "Override with -DCMAKE_CUDA_ARCHITECTURES=<XX>")
    endif()
endif()

# ── Find CUDA Toolkit (gives us CUDA::curand target) ─────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Output directories ────────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/bin)

# ── Create renders directory at configure time ────────────────────────────────
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/renders)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/renders)

# =============================================================================
# Target 1: cpu_raytracer  (plain C++17)
# =============================================================================
add_executable(cpu_raytracer src/cpu_raytracer.cpp)

target_include_directories(cpu_raytracer PRIVATE src)

target_compile_options(cpu_raytracer PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3 /EHsc>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra>
)

# Filesystem support on GCC < 9 requires linking libstdc++fs
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9")
    target_link_libraries(cpu_raytracer PRIVATE stdc++fs)
endif()

# =============================================================================
# Target 2: cuda_raytracer  (CUDA + curand)
# =============================================================================
add_executable(cuda_raytracer src/cuda_raytracer.cu)

target_include_directories(cuda_raytracer PRIVATE src)

target_compile_options(cuda_raytracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --generate-line-info
        -Xcompiler $<IF:$<CXX_COMPILER_ID:MSVC>,/W3,-Wall>
    >
)

target_link_libraries(cuda_raytracer PRIVATE CUDA::curand)

set_target_properties(cuda_raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =============================================================================
# Convenience target: render (builds & runs both)
# =============================================================================
add_custom_target(render
    COMMAND ${CMAKE_BINARY_DIR}/bin/cpu_raytracer
    COMMAND ${CMAKE_BINARY_DIR}/bin/cuda_raytracer
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS cpu_raytracer cuda_raytracer
    COMMENT "Running CPU and GPU renders..."
)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== CUDA Ray Tracer ===")
message(STATUS "Build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures  : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Output directory    : ${CMAKE_BINARY_DIR}/bin")
message(STATUS "")
message(STATUS "After building, run from the project root:")
message(STATUS "  ./build/bin/cpu_raytracer")
message(STATUS "  ./build/bin/cuda_raytracer [--samples 128] [--width 800] [--height 600]")
message(STATUS "")
