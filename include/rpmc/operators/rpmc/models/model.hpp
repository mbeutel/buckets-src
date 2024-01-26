
#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_MODEL_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_MODEL_HPP_


#include <gsl-lite/gsl-lite.hpp>  // for index, type_identity<>, gsl_FailFast()

#include <rpmc/tools/utility.hpp>  // for Span<>

#include <rpmc/operators/rpmc/common.hpp>  // for InteractionRates<>, IsSelfInteraction


namespace rpmc {

namespace gsl = ::gsl_lite;


struct NoOpInteractionModelArgs
{
};

template <template <typename> class TT = std::type_identity_t>
struct NoOpInteractionModelParticleState
{
    friend constexpr auto
    reflect(gsl::type_identity<NoOpInteractionModelParticleState>)
    {
        return makeshift::value_tuple{ };
    }
};
using NoOpInteractionModelState = NoOpInteractionModelParticleState<Span>;

template <template <typename> class TT = std::type_identity_t>
struct NoOpInteractionModelParticleProperties
{
    friend constexpr auto
    reflect(gsl::type_identity<NoOpInteractionModelParticleProperties>)
    {
        return makeshift::value_tuple{ };
    }
};
template <template <typename> class TT = std::type_identity_t>
NoOpInteractionModelParticleProperties<TT>
getParticleProperties(
    NoOpInteractionModelArgs const& /*args*/,
    NoOpInteractionModelParticleState<TT> const& /*ps*/)
{
    return { };
}

class NoOpInteractionModel
{
private:
    NoOpInteractionModelArgs args_;
    NoOpInteractionModelState state_;

public:
    using Locator = DefaultLocator;

    NoOpInteractionModelArgs const&
    getArgs() const
    {
        return args_;
    }
    NoOpInteractionModelState const&
    getState() const
    {
        return state_;
    }

    template <template <typename> class TT = std::type_identity_t>
    struct TracerSwarmInteractionData : InteractionRates<TT>
    {
        TT<double> relativeChange_jk;   // (%)
        TT<double> relativeChange_kj;   // (%)
    };

    void
    initialize()
    {
    }

    void
    synchronize()
    {
    }

    template <template <typename> class TT, typename ParticlePropertiesT>
    TracerSwarmInteractionData<TT>
    computeTracerSwarmInteractionData(
        ParticlePropertiesT const& /*p1*/, ParticlePropertiesT const& /*p2*/,
        TT<IsSelfInteraction> /*isSelfInteraction*/) const
    {
        return {
            {
                .interactionRate_jk = { },
                .interactionRate_kj = { }
            },
            /*.relativeChange_jk =*/ { },
            /*.relativeChange_jk =*/ { }
        };
    }

    template <typename CallbackT, typename ParticlePropertiesT>
    void
    interact(
        CallbackT&& /*callback*/,
        gsl::index /*j*/, gsl::index /*k*/,
        ParticlePropertiesT const& /*p1*/, ParticlePropertiesT const& /*p2*/,
        TracerSwarmInteractionData<> const& /*interactionData*/)
    {
        // callback.invalidate(j);
        // gsl::index k = callback.tryClone(j);
        // callback.swap(j, k);
        gsl_FailFast();
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_MODEL_HPP_
