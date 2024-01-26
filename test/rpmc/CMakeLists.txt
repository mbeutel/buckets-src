
cmake_minimum_required(VERSION 3.20)

find_package(Catch2 3 REQUIRED)

add_executable(test-rpmc
    "test-expandoarray.cpp"
    "test-filling-factors.cpp"
)

target_compile_features(test-rpmc
    PRIVATE
        cxx_std_20
)

target_compile_definitions(test-rpmc
    PRIVATE
        RPMC_TESTING_
        gsl_CONFIG_CONTRACT_VIOLATION_THROWS  # make `gsl_Expects()` throw an exception on precondition failure (so we can write unit tests for failure)
        gsl_CONFIG_CONTRACT_CHECKING_AUDIT  # check `gsl_ExpectsAudit()` preconditions
)

include(TargetCompileSettings)
cmakeshift_target_compile_settings(test-rpmc
    SOURCE_FILE_ENCODING "UTF-8"
)

target_link_libraries(test-rpmc
    PRIVATE
        rpmc
        Catch2::Catch2WithMain
)

add_test(
    NAME test-rpmc
    COMMAND test-rpmc
)
