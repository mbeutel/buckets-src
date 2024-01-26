
#ifndef INCLUDED_RPMC_OPERATORS_RPMC_LOCATORS_LINEAR_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_LOCATORS_LINEAR_HPP_


#include <gsl-lite/gsl-lite.hpp>  // for type_identity<>

#include <rpmc/operators/rpmc/common.hpp>  // for DefaultLocator


namespace rpmc {

namespace gsl = gsl_lite;


struct LinearLocatorParams
{
    double referenceLocation;   // (length)
    double drMin;               // (distance)
    int dsMin = 1;              // number of steps per minimum bin width
};

struct LinearLocatorArgs : LinearLocatorParams
{
    double locationFactor_;

    explicit LinearLocatorArgs(LinearLocatorParams const& params)
        : LinearLocatorParams(params)
    {
        gsl_Expects(drMin > 0);
        gsl_Expects(dsMin >= 1);

        locationFactor_ = dsMin/drMin;
    }
};

template <template <typename> class TT = std::type_identity_t>
struct LinearLocatorParticleState
{
    std::span<double> a;      // position (length)

    friend constexpr auto
    reflect(gsl::type_identity<LinearLocatorParticleState>)
    {
        return makeshift::make_value_tuple(
            makeshift::value_tuple{ &LinearLocatorParticleState::a, "a" }
        );
    }
};
using LinearLocatorState = LinearLocatorParticleState<Span>;

class LinearLocator
{
private:
    LinearLocatorArgs args_;
    LinearLocatorState state_;

public:
    LinearLocator(LinearLocatorParams const& _params, LinearLocatorState const& _state)
        : args_(_params), state_(_state)
    {
    }

    int
    minBinSize() const
    {
        return args_.dsMin;
    }
    double
    location(gsl::index j) const
    {
        double a = state_.a[j];
        return args_.locationFactor_*(a - args_.referenceLocation);
    }
    double
    interactionWidthToDistance(double w) const
    {
        return args_.locationFactor_*w;
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_LOCATORS_LINEAR_HPP_
