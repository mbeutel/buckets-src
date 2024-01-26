
cmake_minimum_required(VERSION 3.24)

find_package(gsl-lite 0.40 REQUIRED)
find_package(fmt REQUIRED)
find_package(makeshift 4 REQUIRED)
find_package(intervals 1.0 REQUIRED)

add_library(rpmc STATIC
    "rpmc.natvis"
    "rpmc.cpp"
)

target_compile_features(rpmc
    PUBLIC
        cxx_std_20
)

if(MSVC)
    # work around MSVC bug
    target_compile_definitions(rpmc
        PUBLIC
            gsl_CONFIG_VALIDATES_UNENFORCED_CONTRACT_EXPRESSIONS=0
    )
endif()

target_include_directories(rpmc
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

include(TargetCompileSettings)
cmakeshift_target_compile_settings(rpmc
    SOURCE_FILE_ENCODING "UTF-8"
)

target_link_libraries(rpmc
    PUBLIC
        gsl::gsl-lite-v1
        fmt::fmt
        makeshift::makeshift
        intervals::intervals
)
