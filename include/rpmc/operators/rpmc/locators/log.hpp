
#ifndef INCLUDED_RPMC_OPERATORS_RPMC_LOCATORS_LOG_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_LOCATORS_LOG_HPP_


#include <gsl-lite/gsl-lite.hpp>  // for type_identity<>

#include <rpmc/operators/rpmc/common.hpp>  // for DefaultLocator


namespace rpmc {

namespace gsl = gsl_lite;


struct LogLocatorParams
{
    double referenceLocation;   // (length)
    double drMin;               // (distance)
    int dsMin = 1;              // number of steps per minimum bin width
};

struct LogLocatorArgs : LogLocatorParams
{
    double rcpReferenceLocation_;
    double locationFactor_;

    explicit LogLocatorArgs(LogLocatorParams const& params)
        : LogLocatorParams(params)
    {
        gsl_Expects(referenceLocation > 0);
        gsl_Expects(drMin > 0);
        gsl_Expects(dsMin >= 1);

        rcpReferenceLocation_ = 1/referenceLocation;
        auto w = drMin/referenceLocation;
        //locationFactor_ = 1/std::log(1 + resolution*minBinWidth*rcpReferenceLocation_);
        auto wSq = w*w;
        auto A = 1 + 1/2*wSq;
        auto B = w*sqrt(1./4*wSq + 1);
        locationFactor_ = dsMin/std::log(A + B);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct LogLocatorParticleState
{
    std::span<double> a;      // position (length)

    friend constexpr auto
    reflect(gsl::type_identity<LogLocatorParticleState>)
    {
        return makeshift::make_value_tuple(
            makeshift::value_tuple{ &LogLocatorParticleState::a, "a" }
        );
    }
};
using LogLocatorState = LogLocatorParticleState<Span>;

class LogLocator
{
private:
    LogLocatorArgs args_;
    LogLocatorState state_;

public:
    LogLocator(LogLocatorParams const& _params, LogLocatorState const& _state)
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
        // Logarithmic coordinates:
        //
        //     ρ = α log r/r₀
        //
        // where  r₀  is the reference location and  α  is the factor
        //
        //     α = [log (1 + Δr/r₀)]⁻¹ ,
        //
        // defined by the condition that  Δr  defines the subclass width,
        //
        //                !
        //     ρ(r₀ + Δr) = 1 .
        //

        double a = state_.a[j];
        return args_.locationFactor_*std::log(a*args_.rcpReferenceLocation_);
    }
    double
    interactionWidthToDistance(double w) const
    {
        // Two particles at locations  r  and  r'  can interact if their distance is less than or equal
        // to their mutual interaction radius  R :
        //
        //     inReach: (ρ, ρ', R) ↦ |r - r'| ≤ R .
        //
        // The interaction range is given implicitly by the `inReach()` predicate:
        //
        //     interactionRange: (r, R) ↦ { r' | inReach(r, r', R) }
        //                              = [r - R, r + R] .
        //
        // If  w = R/√(r⋅r')  is known instead of  R , the interaction range can be determined by
        // solving the equation
        //
        //     |r - r'| = R = √(r⋅r') w
        // 
        // for  r' , obtaining
        //
        //     r' = r [1 + 1/2 w² ± w √(w²/4 + 1)]
        //        ≡ r [  A        ±  B           ]
        //
        // for either end of the range. We thus define the logarithmic distance as
        //
        //     Ρ = α log (A + B) ,
        //
        // noting that, in logarithmic coordinates, additivity is retained:
        //
        //     r + R = r'₊ = r (A + B)
        //  ⇒ ρ(r + R) = α log (r + R)/r₀
        //              = α log r/r₀ + α log (A + B)
        //              = ρ(r) + P ,
        //
        //     r - R = r'₋ = r (A - B)
        //  ⇒ ρ(r - R) = α log (r - R)/r₀
        //              = α log r/r₀ + α log (A - B)
        //              = α log r/r₀ - α log (A - B)⁻¹
        //              = α log r/r₀ - α log [(A + B)/(A² - B²)]
        //              = α log r/r₀ - α log (A + B)                 because A² - B² = 1
        //              = ρ(r) - P .
        //

        if (std::isinf(w))
        {
            return std::numeric_limits<double>::infinity();
        }
        auto wSq = w*w;
        auto A = 1 + 1/2*wSq;
        auto B = w*sqrt(1./4*wSq + 1);
        return args_.locationFactor_*std::log(A + B);
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_LOCATORS_LOG_HPP_
