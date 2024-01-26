
#include <tuple>

#include <gsl-lite/gsl-lite.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <intervals/interval.hpp>

#include <rpmc/detail/filling-factors.hpp>


namespace {

namespace gsl = ::gsl_lite;


TEST_CASE("filling factors")
{
    using namespace intervals::math;
    using intervals::interval;

    SECTION("encloses")
    {
        auto [h1, h2, R] = GENERATE(
            std::tuple{  1.,  1.,   0. },
            std::tuple{  1.,  1.,  1.5 },
            std::tuple{  1.,  1.,  10. },
            std::tuple{  1.,  7.,   0. },
            std::tuple{  1.,  7.,  1.5 },
            std::tuple{  1.,  7.,  10. },
            std::tuple{  4.,  7.,   0. },
            std::tuple{  4.,  7.,  1.5 },
            std::tuple{  4.,  7.,  10. }
        );
        auto da = GENERATE(Catch::Generators::range(-20., 20., 7.));

        auto phi = rpmc::detail::fillingFactor(h1, h2, 0., da, R);
        CAPTURE(phi);

        {
            CAPTURE(h1);
            CAPTURE(h2);
            CAPTURE(R);
            CAPTURE(da);

            auto phiB0 = rpmc::detail::fillingFactor(intervals::interval{ h1 }, intervals::interval{ h2 }, intervals::interval{ 0. }, intervals::interval{ da }, intervals::interval{ R });
            CAPTURE(phiB0);
            CHECK(Catch::Approx(phi) >= phiB0.lower());
            CHECK(Catch::Approx(phi) <= phiB0.upper());
            CHECK(phiB0.lower() == Catch::Approx(phiB0.upper()));
        }

        {
            auto h1w = GENERATE(1., 1.03, 1.3);
            auto h2w = GENERATE(1., 1.07, 1.7);
            auto Rw = GENERATE(1., 1.04, 1.4);
            auto daw = GENERATE(1., 1.04, 2.1);
            auto h1B = intervals::interval{ h1, h1*h1w };
            auto h2B = intervals::interval{ h2, h2*h2w };
            auto RB = intervals::interval{ R, R*Rw };
            auto daB = da < 0 ? intervals::interval{ da, da/daw } : intervals::interval{ da, da*daw };
            CAPTURE(h1B);
            CAPTURE(h2B);
            CAPTURE(RB);
            CAPTURE(daB);
            auto phiB = rpmc::detail::fillingFactor(h1B, h2B, intervals::interval{ 0. }, daB, RB);
            CAPTURE(phiB);
            CHECK(Catch::Approx(phi) >= phiB.lower());
            CHECK(Catch::Approx(phi) <= phiB.upper());
        }
    }
    SECTION("local")
    {
        auto [h1, h2, R] = GENERATE(
            std::tuple{  1.,  8.,  1.5 },
            std::tuple{  3.,  8.,   0. },
            std::tuple{  3.,  8.,  2.5 }
        );
        auto da = GENERATE(-0.5, -0.1, 0., 0.1, 0.5);
        CAPTURE(da);

        auto h1w = GENERATE(1., 1.03, 1.3);
        auto h2w = GENERATE(1., 1.07, 1.7);
        auto Rw = GENERATE(1., 1.04, 1.4);

        auto phi = rpmc::detail::fillingFactor(h1, h2, 0., da, R);
        CAPTURE(phi);

        auto h1B = intervals::interval{ h1, h1*h1w };
        auto h2B = intervals::interval{ h2, h2*h2w };
        auto RB = intervals::interval{ R, R*Rw };
        CAPTURE(h1B);
        CAPTURE(h2B);
        CAPTURE(RB);
        auto phiB = rpmc::detail::fillingFactor(h1B, h2B, intervals::interval{ 0. }, intervals::interval{ da }, RB);
        CAPTURE(phiB);

        auto qB = min(RB/max(h1B, h2B), 1.);
        CHECK(phiB.upper() <= Catch::Approx(qB.upper()));
    }
}


} // anonymous namespace
