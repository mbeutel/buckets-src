
#ifndef INCLUDED_PYRPMC_PYSIMULATION_HPP_
#define INCLUDED_PYRPMC_PYSIMULATION_HPP_


#include <array>

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <gsl-lite/gsl-lite.hpp>  // for type_identity<>, gsl_DEFINE_ENUM_BITMASK_OPERATORS()


namespace py_rpmc {


enum class Effects
{
    none = 0,

        // Viscous stirring as per Ormel et al. (2010).
    stirring   = 0b0001,

        // Dynamical friction as per Ormel et al. (2010).
    friction   = 0b0010,

        // Stochastic collisions.
    collisions = 0b0100,

        // All of the above.
    all        = 0b0111
};
gsl_DEFINE_ENUM_BITMASK_OPERATORS(Effects)
constexpr auto
reflect(gsl_lite::type_identity<Effects>)
{
    return makeshift::value_tuple{
        "Effects",
        "physical effects to consider",
        std::array{
            makeshift::value_tuple{ Effects::none, "none" },
            makeshift::value_tuple{ Effects::stirring, "stirring" },
            makeshift::value_tuple{ Effects::friction, "friction" },
            makeshift::value_tuple{ Effects::collisions, "collisions" },
            makeshift::value_tuple{ Effects::all, "all" }
        }
    };
}


enum class Options
{
    none = 0,

        // Enable locality optimisation.
    locality = 0b1,

        // All of the above.
    all      = 0b1
};
gsl_DEFINE_ENUM_BITMASK_OPERATORS(Options)
constexpr auto
reflect(gsl_lite::type_identity<Options>)
{
    return makeshift::value_tuple{
        "Options",
        "simulation configuration options",
        std::array{
            makeshift::value_tuple{ Options::none, "none" },
            makeshift::value_tuple{ Options::locality, "locality" },
            makeshift::value_tuple{ Options::all, "all" }
        }
    };
}


enum class CollisionKernel
{
        // Constant test kernel.
    constant,

        // Constant test kernel with threshold.
    constantThreshold,

        // Linear test kernel.
    linear,

        // Linear test kernel with threshold.
    linearThreshold,

        // Product test kernel.
    product,

        // Runaway test kernel.
    runaway,

        // Geometric kernel as per Ormel et al. (2010).
    geometric
};
constexpr auto
reflect(gsl_lite::type_identity<CollisionKernel>)
{
    return makeshift::value_tuple{
        "CollisionKernel",
        "collision kernel",
        std::array{
            makeshift::value_tuple{ CollisionKernel::constant, "constant" },
            makeshift::value_tuple{ CollisionKernel::constantThreshold, "constant-threshold" },
            makeshift::value_tuple{ CollisionKernel::linear, "linear" },
            makeshift::value_tuple{ CollisionKernel::linearThreshold, "linear-threshold" },
            makeshift::value_tuple{ CollisionKernel::product, "product" },
            makeshift::value_tuple{ CollisionKernel::runaway, "runaway" },
            makeshift::value_tuple{ CollisionKernel::geometric, "geometric" }
        }
    };
}


} // namespace py_rpmc


#endif // INCLUDED_PYRPMC_PYSIMULATION_HPP_
