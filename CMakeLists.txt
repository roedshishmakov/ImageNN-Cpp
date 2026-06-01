cmake_minimum_required(VERSION 3.16)

project(ImageNNCpp
        VERSION 1.0.0
        DESCRIPTION "Neural network image classifier (C++ port of ImageNN)"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global C++ settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# MSVC: compile sources as UTF-8 so that source files keep working on Windows.
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ---------------------------------------------------------------------------
# Third-party dependencies (fetched and built by CMake)
# ---------------------------------------------------------------------------
include(FetchContent)

# doctest: single-header unit-testing framework.
# SOURCE_SUBDIR points at a non-existent directory on purpose: it makes
# FetchContent only download the sources without adding doctest's own
# CMakeLists.txt (whose cmake_minimum_required is too old for modern CMake).
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    SOURCE_SUBDIR  none
)
FetchContent_MakeAvailable(doctest)

add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${doctest_SOURCE_DIR})
add_library(doctest::doctest ALIAS doctest)

# stb: single-header image loading (replacement for Python Pillow).
# The repository ships no CMakeLists.txt, so it is only populated and then
# exposed through an INTERFACE target pointing at its include directory.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_MakeAvailable(stb)

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Core library: all application logic lives here so that both the executable
# and the tests are built from the very same code.
# ---------------------------------------------------------------------------
add_library(imagenn_core STATIC
    src/version.cpp
    src/activations.cpp
    src/rng.cpp
    src/neuron.cpp
    src/link.cpp
    src/layer.cpp
    src/network.cpp
    src/config.cpp
    src/model_io.cpp
    src/dataset.cpp
    src/plot.cpp
)
target_include_directories(imagenn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(imagenn_core PUBLIC stb_image)
target_compile_features(imagenn_core PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Command-line executable
# ---------------------------------------------------------------------------
add_executable(imagenn src/main.cpp)
target_link_libraries(imagenn PRIVATE imagenn_core)

# ---------------------------------------------------------------------------
# Tests (doctest + ctest)
# ---------------------------------------------------------------------------
enable_testing()

add_executable(imagenn_tests tests/test_main.cpp)
target_link_libraries(imagenn_tests PRIVATE imagenn_core doctest::doctest)
target_compile_definitions(imagenn_tests
    PRIVATE IMAGENN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples")

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(imagenn_tests)
