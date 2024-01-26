
# rpmc


cmake_minimum_required(VERSION 3.24)

project(rpmc
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Define build options.
option(RPMC_BUILD_TESTING "Build tests" OFF)

# We must compile static libraries as position-independent code when making a shared object.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Obtain source dependencies.
include(FetchContent)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        devel
    OVERRIDE_FIND_PACKAGE
)
if(RPMC_BUILD_TESTING)
    # workaround, cf. https://github.com/catchorg/Catch2/issues/2046
    find_package(Catch2 REQUIRED)
    target_compile_features(Catch2 PUBLIC cxx_std_17)
endif()
FetchContent_Declare(gsl-lite
    GIT_REPOSITORY https://github.com/gsl-lite/gsl-lite.git
    GIT_TAG        master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(makeshift
    GIT_REPOSITORY https://github.com/mbeutel/makeshift.git
    GIT_TAG        master
    OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(intervals
    GIT_REPOSITORY https://github.com/mbeutel/intervals.git
    GIT_TAG        master
    OVERRIDE_FIND_PACKAGE
)

# Include target definitions.
add_subdirectory("src/rpmc")
add_subdirectory("src/py-rpmc")
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory("test/rpmc")
endif()
