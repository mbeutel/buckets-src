
// Particle mass classifier for RPMC model.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MERS_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MERS_HPP_


#include <span>
#include <array>
#include <cmath>
#include <utility>     // for pair<>
#include <cstdint>     // for int16_t
#include <optional>
#include <functional>  // for hash<>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, ssize(), type_identity<>

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <intervals/math.hpp>
#include <intervals/interval.hpp>
#include <intervals/type_traits.hpp>

#include <rpmc/operators/rpmc/common.hpp>              // for DefaultRPMCClassifier
#include <rpmc/operators/rpmc/buckets/log.hpp>         // for LogBucketing
#include <rpmc/operators/rpmc/classifiers/regime.hpp>  // for ParticleRegimeClassification

#include <rpmc/const/cgs.hpp>

#include <rpmc/tools/soa.hpp>      // for gatherFromSoA()
#include <rpmc/tools/hash.hpp>     // for hash_combine()
#include <rpmc/tools/utility.hpp>  // for Span<>


namespace rpmc {

namespace gsl = gsl_lite;


enum class BucketExhaustion
{
    none           = 0,
    tracerMass     = 0b0001,
    tracerVelocity = 0b0010,
    tracerPosition = 0b0100,
    tracer         = tracerMass | tracerVelocity | tracerPosition,
    swarm          = 0b1000,
    full           = tracer | swarm
};
gsl_DEFINE_ENUM_BITMASK_OPERATORS(BucketExhaustion)
constexpr auto
reflect(gsl::type_identity<BucketExhaustion>)
{
    return std::array{
        makeshift::value_tuple{ BucketExhaustion::none,           "none"            },
        makeshift::value_tuple{ BucketExhaustion::tracerMass,     "tracer-mass"     },
        makeshift::value_tuple{ BucketExhaustion::tracerVelocity, "tracer-velocity" },
        makeshift::value_tuple{ BucketExhaustion::tracerPosition, "tracer-position" },
        makeshift::value_tuple{ BucketExhaustion::tracer,         "tracer"          },
        makeshift::value_tuple{ BucketExhaustion::swarm,          "swarm"           },
        makeshift::value_tuple{ BucketExhaustion::full,           "full"            }
    };
}


    // mass/eccentricity/radius/swarm regime classifier

struct ParticleMERSRPMCClassifierParams
{
    double MStar;       // mass of the central object                   (g)

    double referenceRadius;  // (length)
    double dr = 0;           // (length)

    double binWideningFraction = 0.05;
    double subclassWideningFraction = 0.05;

    LogBucketingParams MBucketingParams = { .bmin=1, .bmax=1 };
    LogBucketingParams mBucketingParams = { .bmin=1, .bmax=1 };
    LogBucketingParams eBucketingParams = { .bmin=1, .bmax=1, .xmin=1.e-6 };
    LogBucketingParams sinincBucketingParams = { .bmin=1, .bmax=1, .xmin=1.e-6 };

        // In a collision involving a tracer representing `≤ particleRegimeThreshold` particles, it cannot be assumed
        // that the symmetry of collision rates will statistically balance the mass transfer between swarms.
        // Must be  ≥ 1 .
    int particleRegimeThreshold = 100;

    BucketExhaustion exhaustion = BucketExhaustion::none;
};

struct ParticleMERSRPMCClassifierArgs
{
    double MStar;       // mass of the central object                   (g)

    double referenceRadius;  // (length)
    double dr = 0;           // (length)

    double binWideningFraction;
    double subclassWideningFactor;

    LogBucketing MBucketing;
    LogBucketing mBucketing;
    LogBucketing eBucketing;
    LogBucketing sinincBucketing;

        // In a collision involving a tracer representing `≤ particleRegimeThreshold` particles, it cannot be assumed
        // that the symmetry of collision rates will statistically balance the mass transfer between swarms.
        // Must be  ≥ 1 .
    int particleRegimeThreshold = 100;

    BucketExhaustion exhaustion;

    explicit ParticleMERSRPMCClassifierArgs(ParticleMERSRPMCClassifierParams const& params)
        : MStar(params.MStar),
          referenceRadius(params.referenceRadius),
          dr(params.dr),
          binWideningFraction(params.binWideningFraction),
          subclassWideningFactor(1 + params.subclassWideningFraction),
          MBucketing(params.MBucketingParams),
          mBucketing(params.mBucketingParams),
          eBucketing(params.eBucketingParams),
          sinincBucketing(params.sinincBucketingParams),
          exhaustion(params.exhaustion)
    {
        gsl_Expects(params.MStar > 0);
        gsl_Expects(params.referenceRadius > 0);
        gsl_Expects(params.dr >= 0);
        gsl_Expects(params.binWideningFraction >= 0 && params.binWideningFraction < 1);
        gsl_Expects(params.subclassWideningFraction >= 0 && params.subclassWideningFraction < 1);
        gsl_Expects(params.particleRegimeThreshold >= 1);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct ParticleMERSRPMCClassifierParticleState
{
    TT<double> M;       // swarm mass                       (mass)
    TT<double> m;       // particle mass                    (mass)
    TT<double> N;       // number of particles in swarm                     redundant with  M  and  m ; depending on the interaction regime, either  M  or  N  is kept constant
    TT<double> a;       // semimajor axis                   (length)
    TT<double> e;       // eccentricity
    TT<double> sininc;  // inclination

    friend constexpr auto
    reflect(gsl::type_identity<ParticleMERSRPMCClassifierParticleState>)
    {
        return makeshift::make_value_tuple(
            makeshift::value_tuple{ &ParticleMERSRPMCClassifierParticleState::M,      "M"      },
            makeshift::value_tuple{ &ParticleMERSRPMCClassifierParticleState::m,      "m"      },
            makeshift::value_tuple{ &ParticleMERSRPMCClassifierParticleState::N,      "N"      },
            makeshift::value_tuple{ &ParticleMERSRPMCClassifierParticleState::a,      "a"      },
            makeshift::value_tuple{ &ParticleMERSRPMCClassifierParticleState::e,      "e"      },
            makeshift::value_tuple{ &ParticleMERSRPMCClassifierParticleState::sininc, "sininc" }
        );
    }

    //gsl::dim
    //num() const noexcept
    //{
    //    return gsl::ssize(M);  // assuming consistent span lengths
    //}
};
using ParticleMERSRPMCClassifierState = ParticleMERSRPMCClassifierParticleState<Span>;

struct MERSBucketIndex
{
    ParticleRegimeClassification classification;
    std::int8_t rIdx;
    std::int16_t eIdx;
    std::int16_t sinincIdx;
    std::int16_t MIdx;
    std::int16_t mIdx;

    bool operator ==(MERSBucketIndex const& rhs) const = default;
    auto operator <=>(MERSBucketIndex const& rhs) const = default;
};
static_assert(sizeof(MERSBucketIndex) == 10);

class ParticleMERSRPMCClassifier : public DefaultRPMCClassifier
{
private:
    ParticleMERSRPMCClassifierArgs args_;
    ParticleMERSRPMCClassifierState state_;

public:
    ParticleMERSRPMCClassifier(
            ParticleMERSRPMCClassifierParams const& _params,
            ParticleMERSRPMCClassifierState const& _state)
        : args_(_params), state_(_state)
    {
    }

    std::optional<MERSBucketIndex>
    classify(gsl::index j) const
    {
        auto ps = detail::gatherFromSoA(state_, j);

        if (ps.N < 0.9)
        {
            return std::nullopt;
        }

        bool isMany = ps.N > args_.particleRegimeThreshold;
        auto classification = isMany ? ParticleRegimeClassification::manyParticles : ParticleRegimeClassification::fewParticles;

        auto MIdx = gsl::narrow<std::int16_t>(args_.MBucketing.map(ps.M));
        auto mIdx = gsl::narrow<std::int16_t>(args_.mBucketing.map(ps.m));
        auto eIdx = gsl::narrow<std::int16_t>(args_.eBucketing.map(ps.e));
        auto sinincIdx = gsl::narrow<std::int16_t>(args_.sinincBucketing.map(ps.sininc));
        auto rIdx = args_.dr != 0 ? gsl::narrow<std::int8_t>(std::round((ps.a - args_.referenceRadius)/args_.dr)) : std::int8_t{ };
        return MERSBucketIndex{
            .classification = classification,
            .rIdx = rIdx,
            .eIdx = eIdx,
            .sinincIdx = sinincIdx,
            .MIdx = MIdx,
            .mIdx = mIdx
        };
    }

    template <template <template <typename> class> class ParticleStateT, template <typename> class TT>
    ParticleStateT<intervals::set_of_t>
    widen(ParticleStateT<TT> const& particleState) const
    {
        using namespace intervals::math;
        using namespace intervals::logic;

            // Scalars (or intervals) to intervals.
        auto result = makeshift::apply(
            [](auto const&... members)
            {
                return ParticleStateT<intervals::set_of_t>{ members... };
            },
            makeshift::tie_members(particleState));

        auto newm = args_.mBucketing.widen(result.m, args_.binWideningFraction);
        if (always(result.N > args_.particleRegimeThreshold))  // strictly many-particles swarms
        {
            auto newM = result.M;
            auto newN = newM/newm;
            if (always(newN > args_.particleRegimeThreshold))  // still strictly many-particles swarms
            {
                reset(result.m, newm);
                reset(result.N, newN);
            }
        }
        else if (always(result.N <= args_.particleRegimeThreshold))  // strictly few-particles swarms
        {
                // Widen  M  and  m  together but leave  N  unwidened.
            auto newM = args_.mBucketing.widen(result.M, args_.binWideningFraction);
            reset(result.M, newM);
            reset(result.m, newm);
        }
        // if either kind of swarm is included, don't widen (shouldn't happen)

        reset(result.a, rpmc::widenAdditively(result.a, args_.binWideningFraction*args_.dr));
        reset(result.e, args_.eBucketing.widen(result.e, args_.binWideningFraction));
        reset(result.sininc, args_.sinincBucketing.widen(result.sininc, args_.binWideningFraction));

        return result;
    }

    bool
    isActive(gsl::index j) const
    {
        double N = state_.N[j];
        return N > 0.9;
    }

    double
    widenInteractionWidth(double w) const
    {
        return w*args_.subclassWideningFactor;
    }
};


} // namespace rpmc


template <>
struct std::hash<rpmc::MERSBucketIndex>
{
    std::size_t
    operator ()(rpmc::MERSBucketIndex const& x) const noexcept
    {
        std::size_t seed = 0;
        rpmc::detail::hash_combine_old(seed, static_cast<std::uint16_t>(x.classification));
        rpmc::detail::hash_combine_old(seed, x.rIdx);
        rpmc::detail::hash_combine_old(seed, x.eIdx);
        rpmc::detail::hash_combine_old(seed, x.sinincIdx);
        rpmc::detail::hash_combine_old(seed, x.MIdx);
        rpmc::detail::hash_combine_old(seed, x.mIdx);
        return seed;
    }
};


#endif // INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_MERS_HPP_
