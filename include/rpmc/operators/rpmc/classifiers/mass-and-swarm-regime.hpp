
// Particle mass classifier for RPMC model.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MASSANDSWARMREGIME_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MASSANDSWARMREGIME_HPP_


#include <span>
#include <array>
#include <compare>
#include <utility>     // for pair<>
#include <cstdint>     // for int16_t
#include <optional>
#include <functional>  // for hash<>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, ssize(), type_identity<>

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <intervals/interval.hpp>
#include <intervals/type_traits.hpp>

#include <rpmc/operators/rpmc/common.hpp>              // for DefaultRPMCClassifier
#include <rpmc/operators/rpmc/classifiers/regime.hpp>  // for ParticleRegimeClassification

#include <rpmc/tools/hash.hpp>     // for hash_combine()
#include <rpmc/tools/utility.hpp>  // for Span<>


namespace rpmc {

namespace gsl = gsl_lite;


struct ParticleMassAndSwarmRegimeRPMCClassifierParams
{
    double referenceMass;   // (mass)

    LogBucketingParams MBucketingParams = { .bmin=2, .bmax=2 };
    LogBucketingParams mBucketingParams = { .bmin=2, .bmax=2 };

        // In a collision involving a tracer representing `≤ particleRegimeThreshold` particles, it cannot be assumed
        // that the symmetry of collision rates will statistically balance the mass transfer between swarms.
        // Must be  ≥ 1 .
    int particleRegimeThreshold = 100;
};

struct ParticleMassAndSwarmRegimeRPMCClassifierArgs
{
    double referenceMass;   // (mass)

    LogBucketing MBucketing;
    LogBucketing mBucketing;

    int particleRegimeThreshold;

    explicit ParticleMassAndSwarmRegimeRPMCClassifierArgs(ParticleMassAndSwarmRegimeRPMCClassifierParams const& params)
        : referenceMass(params.referenceMass),
          MBucketing(params.MBucketingParams),
          mBucketing(params.mBucketingParams),
          particleRegimeThreshold(params.particleRegimeThreshold)
    {
        gsl_Expects(params.referenceMass > 0);
        gsl_Expects(params.particleRegimeThreshold >= 1);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct ParticleMassAndSwarmRegimeRPMCClassifierParticleState
{
    std::span<double> M;      // swarm mass     (mass)
    std::span<double> m;      // particle mass  (mass)

    friend constexpr auto
    reflect(gsl::type_identity<ParticleMassAndSwarmRegimeRPMCClassifierParticleState>)
    {
        return makeshift::make_value_tuple(
            makeshift::value_tuple{ &ParticleMassAndSwarmRegimeRPMCClassifierParticleState::M, "M" },
            makeshift::value_tuple{ &ParticleMassAndSwarmRegimeRPMCClassifierParticleState::m, "m" }
        );
    }
};
using ParticleMassAndSwarmRegimeRPMCClassifierState = ParticleMassAndSwarmRegimeRPMCClassifierParticleState<Span>;


struct ParticleMassAndSwarmRegimeBucketIndex
{
    ParticleRegimeClassification classification;
    std::int16_t MIdx;
    std::int16_t mIdx;

    bool operator ==(ParticleMassAndSwarmRegimeBucketIndex const& rhs) const = default;
    auto operator <=>(ParticleMassAndSwarmRegimeBucketIndex const& rhs) const = default;
};

class ParticleMassAndSwarmRegimeRPMCClassifier : public DefaultRPMCClassifier
{
private:
    ParticleMassAndSwarmRegimeRPMCClassifierArgs args_;
    ParticleMassAndSwarmRegimeRPMCClassifierState state_;

public:
    ParticleMassAndSwarmRegimeRPMCClassifier(ParticleMassAndSwarmRegimeRPMCClassifierParams const& _params, ParticleMassAndSwarmRegimeRPMCClassifierState const& _state)
        : args_(_params), state_(_state)
    {
    }

    std::optional<ParticleMassAndSwarmRegimeBucketIndex>
    classify(gsl::index j) const
    {
            // We check for  M > 0.9 m , not  M ≥ 0.9 m ,  to make sure the condition evaluates to `false` if  M = m = 0 .
        double M = state_.M[j];
        double m = state_.m[j];
        if (M <= 0.9*m)
        {
            return std::nullopt;
        }

            // We check for  M > qₘ m , not  M ≥ qₘ m ,  to make sure the condition evaluates to `false` if  M = m = 0 .
        bool isMany = M > args_.particleRegimeThreshold*m;
        auto classification = isMany ? ParticleRegimeClassification::manyParticles : ParticleRegimeClassification::fewParticles;
        auto MIdx = gsl::narrow<std::int16_t>(args_.MBucketing.map(M));
        auto mIdx = gsl::narrow<std::int16_t>(args_.mBucketing.map(m));
        return ParticleMassAndSwarmRegimeBucketIndex{
            .classification = classification,
            .MIdx = MIdx,
            .mIdx = mIdx
        };
    }

    bool
    isActive(gsl::index j) const
    {
            // We check for  M > 0.9 m , not  M ≥ 0.9 m ,  to make sure the condition evaluates to `false` if  M = m = 0 .
        double M = state_.M[j];
        double m = state_.m[j];
        return M > 0.9*m;
    }
};


} // namespace rpmc


template <>
struct std::hash<rpmc::ParticleMassAndSwarmRegimeBucketIndex>
{
    constexpr std::size_t
    operator ()(rpmc::ParticleMassAndSwarmRegimeBucketIndex const& x) const noexcept
    {
        std::size_t seed = 0;
        rpmc::detail::hash_combine_old(seed, static_cast<std::uint16_t>(x.classification));
        rpmc::detail::hash_combine_old(seed, x.MIdx);
        rpmc::detail::hash_combine_old(seed, x.mIdx);
        return seed;
    }
};


#endif // INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MASSANDSWARMREGIME_HPP_
