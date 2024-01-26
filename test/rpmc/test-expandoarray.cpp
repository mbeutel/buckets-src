
#include <gsl-lite/gsl-lite.hpp>  // for fail_fast

#include <catch2/catch_test_macros.hpp>

#include <rpmc/tools/expandoarray.hpp>


namespace {

namespace gsl = ::gsl_lite;


TEST_CASE("ExpandoArray<>")
{
    SECTION("0d")
    {
        auto a = rpmc::ExpandoArray<int, 0>{ };
        CHECK(a[{ }] == 0);
        a.assign({ }, 1);
        CHECK(a[{ }] == 1);
    }
    SECTION("1d")
    {
        auto a = rpmc::ExpandoArray<int, 1>{ };
        CHECK_THROWS_AS(a[{ 3 }], gsl::fail_fast);
        a.assign({ 3 }, 1);
        a.assign({ -1 }, 2);
        CHECK(a[{ 3 }] == 1);
        CHECK(a[{ 2 }] == 0);
        CHECK(a[{ 0 }] == 0);
        CHECK(a[{ -1 }] == 2);
    }
    SECTION("1d-fixed")
    {
        auto a = rpmc::ExpandoArray<int, 1>{ { 2 } };
        CHECK(a[{ 0 }] == 0);
        CHECK(a[{ 1 }] == 0);
        a.assign({ 0 }, 2);
        a.assign({ 1 }, 1);
        CHECK(a[{ 1 }] == 1);
        CHECK(a[{ 0 }] == 2);
        CHECK_THROWS_AS(a[{ -1 }], gsl::fail_fast);
        CHECK_THROWS_AS(a[{ 2 }], gsl::fail_fast);
    }
    SECTION("2d")
    {
        auto a = rpmc::ExpandoArray<int, 2>{ };
        CHECK_THROWS_AS((a[{ -4, 5 }]), gsl::fail_fast);
        a.assign({ -4, 5 }, 1);
        CHECK(a[{ -4, 5 }] == 1);
        a.assign({ 3, 6 }, 2);
        CHECK(a[{ -4, 5 }] == 1);
        CHECK(a[{ 3, 6 }] == 2);
        a.assign({ -8, 12 }, 3);
        CHECK(a[{ -4, 5 }] == 1);
        CHECK(a[{ 3, 6 }] == 2);
        CHECK(a[{ -8, 12 }] == 3);
    }
    SECTION("2d-fixed-y")
    {
        auto a = rpmc::ExpandoArray<int, 2>{ { -1, 7 } };
        CHECK_THROWS_AS((a[{ -4, 5 }]), gsl::fail_fast);
        a.assign({ -4, 5 }, 1);
        CHECK(a[{ -4, 5 }] == 1);
        a.assign({ 3, 6 }, 2);
        CHECK(a[{ -4, 5 }] == 1);
        CHECK(a[{ 3, 6 }] == 2);
        CHECK_THROWS_AS((a[{ 3, 7 }]), gsl::fail_fast);
        CHECK_THROWS_AS((a[{ 3, -1 }]), gsl::fail_fast);
        CHECK_THROWS_AS((a[{ -8, 7 }]), gsl::fail_fast);
        CHECK_THROWS_AS((a[{ -8, -1 }]), gsl::fail_fast);
    }
}


} // anonymous namespace
