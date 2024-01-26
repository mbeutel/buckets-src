
cmake_minimum_required(VERSION 3.24)

find_package(gsl-lite 0.40 REQUIRED)
find_package(fmt REQUIRED)
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(makeshift 4 REQUIRED)
find_package(intervals 1.0 REQUIRED)

pybind11_add_module(py-rpmc
    "assert-handler.cpp"
    "module.cpp"
    "py-convert.cpp"
    "py-rpmc-simulation.cpp"
    "py-rpmc-traditional-simulation.cpp"
)

target_compile_features(py-rpmc
    PRIVATE
        cxx_std_20
)

target_precompile_headers(py-rpmc
    PRIVATE
        <cmath>
        <array>
        <tuple>
        <string>
        <vector>
        <random>
        <variant>
        <cassert>
        <utility>
        <cstddef>
        <cstdint>
        <numeric>
        <optional>
        <stdexcept>
        <algorithm>
        <functional>
        <string_view>
        <fmt/core.h>
        <pybind11/pybind11.h>
        <pybind11/numpy.h>
        <pybind11/stl.h>
)

set_target_properties(py-rpmc
    PROPERTIES
        OUTPUT_NAME "rpmc"  # this is the module name seen by Python
)

target_include_directories(py-rpmc
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Make sure the Python module ends up in the build directory and not in some subdirectory of it.
set_target_properties(py-rpmc
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
        VS_JUST_MY_CODE_DEBUGGING "$<IF:$<CONFIG:Debug>,ON,OFF>"
)

include(TargetCompileSettings)
cmakeshift_target_compile_settings(py-rpmc
    SOURCE_FILE_ENCODING "UTF-8"
    #FP_MODEL "fastest"
    #SANITIZE "address" "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:undefined>"
)

# Make `gsl_Expects()` throw an exception on precondition failure (`terminate()` is inconvenient in Python REPL).
target_compile_definitions(py-rpmc
    PRIVATE
        gsl_CONFIG_CONTRACT_VIOLATION_CALLS_HANDLER
        #"$<$<CONFIG:Debug>:gsl_CONFIG_CONTRACT_CHECKING_AUDIT>"  # also enable audit-level checks in Debug mode
        #gsl_CONFIG_CONTRACT_CHECKING_AUDIT  # also enable audit-level checks
)

if(MSVC)
    target_compile_options(py-rpmc
        PRIVATE
            "/bigobj"  # enable /bigobj switch to permit more than 2^16 COMDAT sections per .obj file (can be useful in heavily templatized code)
            "/Zc:inline"  # remove unreferenced COMDATs to improve linker throughput
            "/Zc:__cplusplus"  # have `__cplusplus` report the correct language version
            #"/Zc:externConstexpr"  # enable conforming behavior and use external linkage for `extern constexpr` variables  # TODO: not recognized by clang-cl
            "/volatile:iso"  # use strict ISO C++ `volatile` semantics
            "/permissive-"  # enable non-permissive mode
    )
endif()

target_link_libraries(py-rpmc
    PRIVATE
        rpmc
        gsl::gsl-lite-v1
        fmt::fmt
        makeshift::makeshift
        intervals::intervals
)
