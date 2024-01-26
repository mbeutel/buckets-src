
#ifndef INCLUDED_RPMC_OPERATORS_RPMC_COMMON_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_COMMON_HPP_


#include <array>

#include <gsl-lite/gsl-lite.hpp>  // for type_identity<>

#include <makeshift/tuple.hpp>

#include <intervals/type_traits.hpp>  // for set_of_t<>


namespace rpmc {

namespace gsl = gsl_lite;


enum class IsSelfInteraction : bool { no = false, yes = true };
constexpr auto reflect(gsl::type_identity<IsSelfInteraction>)
{
    return std::array{ IsSelfInteraction::no, IsSelfInteraction::yes };
}

template <template <typename> class TT = std::type_identity_t>
struct InteractionRates
{
    TT<double> interactionRate_jk;  // (time⁻¹)
    TT<double> interactionRate_kj;  // (time⁻¹)
};
template <template <typename> class TT = std::type_identity_t>
struct LocalInteractionRates : InteractionRates<TT>
{
    TT<double> interactionWidth;  // (length)
};


class DefaultRPMCBaseClassifier
{
public:
    static constexpr bool haveLocality = false;

    bool
    isActive(gsl::index /*j*/) const
    {
        return true;
    }
};

class DefaultRPMCClassifier : public DefaultRPMCBaseClassifier
{
public:
    std::array<std::int16_t, 0>
    classify(gsl::index /*j*/) const
    {
        return { };
    }

    template <template <template <typename> class> class ParticleStateT, template <typename> class TT>
    ParticleStateT<intervals::set_of_t>
    widen(ParticleStateT<TT> const& particleState) const
    {
            // Scalars (or intervals) to intervals.
        return makeshift::apply(
            [](auto const&... members)
            {
                return ParticleStateT<intervals::set_of_t>{ members... };
            },
            makeshift::tie_members(particleState));
    }

    double
    widenInteractionWidth(double w) const
    {
        return w;
    }
};


class DefaultLocator
{
public:
    int
    minBinSize() const
    {
        return 1;
    }
    double
    location(gsl::index /*j*/) const
    {
        return { };
    }
    double
    interactionWidthToDistance(double w) const
    {
        return w;
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_COMMON_HPP_
