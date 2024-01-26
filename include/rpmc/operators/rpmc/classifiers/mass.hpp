
// Particle mass classifier for RPMC model.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MASS_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MASS_HPP_


#include <span>
#include <array>
#include <cstdint>  // for int16_t

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, ssize(), type_identity<>

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <rpmc/operators/rpmc/common.hpp>  // for DefaultRPMCBaseClassifier, DefaultRPMCClassifier

#include <rpmc/tools/utility.hpp>  // for Span<>


namespace rpmc {

namespace gsl = gsl_lite;


struct ParticleMassRPMCClassifierParams
{
    double referenceMass;   // (mass)
};

struct ParticleMassRPMCClassifierArgs : ParticleMassRPMCClassifierParams
{
    explicit ParticleMassRPMCClassifierArgs(ParticleMassRPMCClassifierParams const& params)
        : ParticleMassRPMCClassifierParams(params)
    {
        gsl_Expects(params.referenceMass > 0);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct ParticleMassRPMCClassifierParticleState
{
    std::span<double> M;      // swarm mass     (mass)
    std::span<double> m;      // particle mass  (mass)

    friend constexpr auto
    reflect(gsl::type_identity<ParticleMassRPMCClassifierParticleState>)
    {
        return makeshift::make_value_tuple(
            makeshift::value_tuple{ &ParticleMassRPMCClassifierParticleState::M, "M" },
            makeshift::value_tuple{ &ParticleMassRPMCClassifierParticleState::m, "m" }
        );
    }
};
using ParticleMassRPMCClassifierState = ParticleMassRPMCClassifierParticleState<Span>;

class ParticleMassRPMCBaseClassifier : public DefaultRPMCBaseClassifier
{
private:
    ParticleMassRPMCClassifierState state_;

public:
    ParticleMassRPMCBaseClassifier(ParticleMassRPMCClassifierState const& _state)
        : state_(_state)
    {
    }

    bool
    isActive(gsl::index j) const
    {
            // We check for  M > 0.9 m , not  M ≥ 0.9 m ,  to make sure the condition evaluates to `false` if  M = m = 0 .
        return state_.M[j] > 0.9*state_.m[j];
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MASS_HPP_
